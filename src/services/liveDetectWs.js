import { WebSocketServer } from 'ws';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { INFERENCE_URL } from '../config.js';
import { predict } from './modelService.js';

/**
 * Setup WebSocket Server on Express HTTP server for live detection streaming
 * Route: /live-detect
 */
export function setupLiveDetectWebSocket(server) {
  const wss = new WebSocketServer({ server, path: '/live-detect' });

  wss.on('connection', (ws, req) => {
    const clientIp = req.socket.remoteAddress;
    console.log(`📡 [LiveDetectWS] Client connected from ${clientIp}`);

    ws.on('message', async (chunk) => {
      if (!chunk || chunk.length === 0) return;

      const chunkKb = (chunk.length / 1024).toFixed(1);
      console.log(`🎤 [LiveDetectWS] Received audio chunk (${chunkKb} KB) from client`);

      try {
        let score = null;
        let sourceEngine = 'none';

        // 1. First attempt: Forward raw audio chunk to Python inference service (/infer)
        if (INFERENCE_URL) {
          try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 6000);
            const res = await fetch(`${INFERENCE_URL}/infer`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: chunk,
              signal: controller.signal,
            });
            clearTimeout(timer);

            if (res.ok) {
              const data = await res.json();
              if (typeof data.score === 'number') {
                score = data.score;
                sourceEngine = 'remote_fastapi_infer';
              } else if (typeof data.spoof_prob === 'number') {
                score = data.spoof_prob;
                sourceEngine = 'remote_fastapi_infer';
              }
            }
          } catch (inferErr) {
            // Service might be using /predict-file or local fallback
            console.debug('[LiveDetectWS] /infer endpoint attempt:', inferErr.message);
          }
        }

        // 2. Fallback: Save chunk to temporary file and run through prediction engine
        if (score === null) {
          const tempDir = os.tmpdir();
          const isWav = Buffer.isBuffer(chunk) && chunk.toString('utf8', 0, 4) === 'RIFF';
          const isWebm = Buffer.isBuffer(chunk) && chunk[0] === 0x1a && chunk[1] === 0x45;
          const ext = isWav ? '.wav' : isWebm ? '.webm' : '.wav';
          const tempFile = path.join(tempDir, `live_chunk_${Date.now()}_${Math.random().toString(36).slice(2, 6)}${ext}`);

          try {
            // If raw Float32 PCM (from AudioWorklet), wrap into a basic WAV header
            let bufferToWrite = chunk;
            if (!isWav && !isWebm && chunk.length % 4 === 0) {
              bufferToWrite = pcmFloat32ToWavBuffer(chunk, 48000);
            }

            fs.writeFileSync(tempFile, bufferToWrite);
            const pred = await predict(tempFile, { fastMode: true });
            if (pred) {
              sourceEngine = pred.engine || 'local_model';
              score = typeof pred.spoof_prob === 'number'
                ? pred.spoof_prob
                : pred.label === 'cloned'
                  ? pred.confidence
                  : 1.0 - pred.confidence;
            }
          } finally {
            try {
              if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
            } catch (_) {}
          }
        }

        if (score !== null && !isNaN(score)) {
          // Clamp score to 0.0 - 1.0
          const clampedScore = Math.max(0, Math.min(1, Math.round(score * 1000) / 1000));
          const verdict = clampedScore >= 0.5 ? '🔴 LIKELY CLONED / SYNTHETIC' : '🟢 LIKELY AUTHENTIC HUMAN';
          console.log(`📊 [LiveDetectWS] Live Prediction: ${(clampedScore * 100).toFixed(1)}% cloned | Verdict: ${verdict} | Engine: ${sourceEngine}`);

          if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify({
              score: clampedScore,
              verdict: clampedScore >= 0.5 ? 'cloned' : 'human',
              engine: sourceEngine,
              timestamp: Date.now()
            }));
          }
        } else {
          console.warn('⚠️ [LiveDetectWS] Unable to obtain inference score');
          if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify({ error: 'inference_failed' }));
          }
        }
      } catch (err) {
        console.error('❌ [LiveDetectWS] Inference error:', err.message);
        if (ws.readyState === ws.OPEN) {
          ws.send(JSON.stringify({ error: 'inference_failed' }));
        }
      }
    });

    ws.on('close', () => {
      console.log(`🔌 [LiveDetectWS] Client disconnected (${clientIp})`);
    });

    ws.on('error', (err) => {
      console.warn(`⚠️ [LiveDetectWS] Socket error:`, err.message);
    });
  });

  return wss;
}

/**
 * Helper to wrap raw Float32Array PCM byte buffer into 16-bit PCM WAV Buffer
 */
function pcmFloat32ToWavBuffer(rawBuffer, sampleRate = 48000) {
  const numSamples = Math.floor(rawBuffer.length / 4);
  const wavHeaderSize = 44;
  const pcm16ByteLength = numSamples * 2;
  const outBuf = Buffer.alloc(wavHeaderSize + pcm16ByteLength);

  // RIFF header
  outBuf.write('RIFF', 0);
  outBuf.writeUInt32LE(36 + pcm16ByteLength, 4);
  outBuf.write('WAVE', 8);
  outBuf.write('fmt ', 12);
  outBuf.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
  outBuf.writeUInt16LE(1, 20); // AudioFormat (1 = PCM)
  outBuf.writeUInt16LE(1, 22); // NumChannels (1)
  outBuf.writeUInt32LE(sampleRate, 24); // SampleRate
  outBuf.writeUInt32LE(sampleRate * 2, 28); // ByteRate
  outBuf.writeUInt16LE(2, 32); // BlockAlign
  outBuf.writeUInt16LE(16, 34); // BitsPerSample
  outBuf.write('data', 36);
  outBuf.writeUInt32LE(pcm16ByteLength, 40);

  // Convert float samples to int16
  for (let i = 0; i < numSamples; i++) {
    const floatVal = rawBuffer.readFloatLE(i * 4);
    const clamped = Math.max(-1, Math.min(1, floatVal));
    const int16Val = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    outBuf.writeInt16LE(Math.round(int16Val), wavHeaderSize + i * 2);
  }

  return outBuf;
}

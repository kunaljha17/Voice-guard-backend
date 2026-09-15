import http from 'http';
import express from 'express';
import WebSocket from 'ws';
import { setupLiveDetectWebSocket } from '../services/liveDetectWs.js';

async function runLiveDetectionTest() {
  console.log('════════════════════════════════════════════════════════════');
  console.log('🧪 VoiceGuard Live Detection WebSocket Verification Test');
  console.log('════════════════════════════════════════════════════════════');

  // 1. Spin up a test HTTP server on port 5999 to test without colliding with port 5000
  const app = express();
  const testServer = http.createServer(app);
  const testPort = 5999;

  setupLiveDetectWebSocket(testServer);

  await new Promise((resolve) => {
    testServer.listen(testPort, '127.0.0.1', () => {
      console.log(`[TestServer] Live WebSocket listening on ws://127.0.0.1:${testPort}/live-detect`);
      resolve();
    });
  });

  // 2. Connect client WebSocket
  const wsUrl = `ws://127.0.0.1:${testPort}/live-detect`;
  const clientWs = new WebSocket(wsUrl);

  const receivedScores = [];

  await new Promise((resolve, reject) => {
    clientWs.on('open', async () => {
      console.log('✅ Client WebSocket connected successfully!');

      // Generate 3 seconds of 48kHz Float32 raw speech-like PCM audio
      const sampleRate = 48000;
      const durationSec = 3;
      const numSamples = sampleRate * durationSec;
      const float32Array = new Float32Array(numSamples);

      // Synthesize authentic speech-like chunk
      for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const f0 = 130 + Math.sin(2 * Math.PI * 3 * t) * 15;
        const wave =
          Math.sin(2 * Math.PI * f0 * t) * 0.4 +
          Math.sin(2 * Math.PI * f0 * 2 * t) * 0.2 +
          Math.sin(2 * Math.PI * f0 * 3 * t) * 0.1;
        float32Array[i] = wave * (0.5 + 0.5 * Math.sin(2 * Math.PI * 1.5 * t));
      }

      const humanChunk = Buffer.from(float32Array.buffer);
      console.log(`📤 [Test 1] Sending authentic voice chunk (${(humanChunk.length / 1024).toFixed(1)} KB, Float32 PCM)...`);
      clientWs.send(humanChunk);

      // Synthesize synthetic vocoded robot audio chunk
      const clonedArray = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate;
        const f0 = 160; // rigid flat pitch
        let wave = Math.sin(2 * Math.PI * f0 * t) * 0.5;
        // Vocoder quantization
        wave = Math.round(wave * 4) / 4;
        clonedArray[i] = wave;
      }
      const clonedChunk = Buffer.from(clonedArray.buffer);

      setTimeout(() => {
        console.log(`📤 [Test 2] Sending synthetic vocoded voice chunk (${(clonedChunk.length / 1024).toFixed(1)} KB, Float32 PCM)...`);
        clientWs.send(clonedChunk);
      }, 400);
    });

    clientWs.on('message', (data) => {
      try {
        const parsed = JSON.parse(data.toString());
        console.log('📥 [WebSocket Client Received]:', JSON.stringify(parsed, null, 2));
        receivedScores.push(parsed);

        if (receivedScores.length >= 2) {
          clientWs.close();
          testServer.close(() => {
            resolve();
          });
        }
      } catch (err) {
        console.error('Failed to parse message:', err);
      }
    });

    clientWs.on('error', (err) => {
      console.error('❌ WebSocket Client Error:', err);
      clientWs.close();
      testServer.close();
      reject(err);
    });

    // Timeout after 15 seconds
    setTimeout(() => {
      if (receivedScores.length === 0) {
        clientWs.close();
        testServer.close();
        reject(new Error('Live detection test timed out waiting for responses'));
      } else {
        clientWs.close();
        testServer.close(() => resolve());
      }
    }, 15000);
  });

  console.log('────────────────────────────────────────────────────────────');
  console.log(`✨ Test Summary: Received ${receivedScores.length} live predictions.`);
  for (const [idx, item] of receivedScores.entries()) {
    const pct = typeof item.score === 'number' ? (item.score * 100).toFixed(1) : 'N/A';
    console.log(
      `   Chunk #${idx + 1}: Score = ${pct}% likely cloned | Verdict = ${item.verdict || 'N/A'} | Engine = ${item.engine || 'N/A'}`
    );
  }
  console.log('════════════════════════════════════════════════════════════');
}

runLiveDetectionTest()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('Test failed:', err);
    process.exit(1);
  });

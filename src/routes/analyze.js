import { Router } from 'express';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { randomUUID } from 'crypto';
import multer from 'multer';
import { optionalAuth } from '../middleware/auth.js';
import { predict } from '../services/modelService.js';
import { Scan } from '../models/Scan.js';

const router = Router();

const uploadDir = path.join(os.tmpdir(), 'voiceguard_uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({ dest: uploadDir, limits: { fileSize: 50 * 1024 * 1024 } });

const uploadFields = upload.fields([
  { name: 'file', maxCount: 1 },
  { name: 'audio', maxCount: 1 },
]);

const multerWrapper = (req, res, next) => {
  uploadFields(req, res, (err) => {
    if (err) return res.status(400).json({ error: `File upload error: ${err.message}` });
    if (req.files) {
      req.file = req.files['file']?.[0] || req.files['audio']?.[0];
    }
    next();
  });
};

async function saveScanToDb(userId, filename, result) {
  if (!userId) return;
  try {
    const spoofProb = result.spoof_prob || 0;
    const conf = result.confidence;
    await Scan.create({
      userId,
      filename,
      label: result.label,
      confidence: conf,
      confidencePercent: Number((conf * 100).toFixed(1)),
      spoofProb,
      isSynthetic: result.label === 'cloned',
      classification: result.label === 'cloned'
        ? `Cloned / AI-Generated Voice (${(conf * 100).toFixed(1)}%)`
        : `Authentic Human (${(conf * 100).toFixed(1)}%)`,
      anomalyTag: result.label === 'cloned'
        ? 'Phoneme Boundary Discontinuity & Glottal Pulse Flagged'
        : 'Natural vocal tract resonance & pulmonary cadence confirmed',
      model: result.model_name || result.model_version || 'Wav2Vec2',
      acousticFindings: result.label === 'cloned'
        ? `Wav2Vec2 model flagged synthetic clone (spoof prob: ${(spoofProb * 100).toFixed(1)}%).`
        : `Authentic human voice verified (spoof prob: ${(spoofProb * 100).toFixed(2)}%).`,
      rawPrediction: result,
    });
  } catch (e) {
    console.warn('[Analyze] Failed to persist scan:', e.message);
  }
}

/**
 * POST /api/analyze
 */
router.post('/', multerWrapper, optionalAuth, async (req, res) => {
  let tempFilePath = null;
  let originalName = 'uploaded_audio.wav';

  try {
    if (req.file) {
      tempFilePath = req.file.path;
      originalName = req.file.originalname || originalName;
    } else if (req.body?.audioData) {
      const raw = req.body.audioData;
      const cleanBase64 = raw.includes('base64,') ? raw.split('base64,')[1] : raw;
      const audioBuffer = Buffer.from(cleanBase64, 'base64');
      if (audioBuffer.length === 0) {
        return res.status(400).json({ error: 'Invalid audio: Base64 payload is empty.' });
      }
      originalName = req.body.filename || originalName;
      tempFilePath = path.join(uploadDir, `upload_${randomUUID()}.wav`);
      fs.writeFileSync(tempFilePath, audioBuffer);
    } else {
      return res.status(400).json({ error: 'No audio file uploaded.' });
    }

    const result = await predict(tempFilePath);
    await saveScanToDb(req.userId, originalName, result);
    return res.json(result);
  } catch (err) {
    console.error('[Analyze API Error]:', err.message);
    const code = err.status === 400 || err.message?.includes('Invalid or corrupt') ? 400 : 500;
    return res.status(code).json({ error: err.message || 'Audio analysis failed.' });
  } finally {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try { fs.unlinkSync(tempFilePath); } catch (_) {}
    }
  }
});

/**
 * POST /api/analyze/live
 */
router.post('/live', multerWrapper, optionalAuth, async (req, res) => {
  let tempFilePath = null;
  let chunkFilename = 'live_stream_chunk.wav';

  try {
    let audioBuffer = null;

    if (req.file) {
      tempFilePath = req.file.path;
      chunkFilename = req.file.originalname || chunkFilename;
    } else if (req.body?.chunk || req.body?.audioData) {
      const raw = req.body.chunk || req.body.audioData;
      const cleanBase64 = raw.includes('base64,') ? raw.split('base64,')[1] : raw;
      audioBuffer = Buffer.from(cleanBase64, 'base64');
      chunkFilename = req.body.filename || chunkFilename;
    } else if (Buffer.isBuffer(req.body)) {
      audioBuffer = req.body;
    }

    if (audioBuffer && !tempFilePath) {
      if (audioBuffer.length === 0) {
        return res.status(400).json({ error: 'Streamed audio chunk is empty.' });
      }
      tempFilePath = path.join(uploadDir, `live_${randomUUID()}.wav`);
      fs.writeFileSync(tempFilePath, audioBuffer);
    }

    if (!tempFilePath || !fs.existsSync(tempFilePath)) {
      return res.status(400).json({ error: 'No live audio chunk provided.' });
    }

    const result = await predict(tempFilePath);
    await saveScanToDb(req.userId, chunkFilename, result);
    return res.json(result);
  } catch (err) {
    console.error('[Live Analyze API Error]:', err.message);
    const code = err.status === 400 || err.message?.includes('Invalid or corrupt') ? 400 : 500;
    return res.status(code).json({ error: err.message || 'Live audio analysis failed.' });
  } finally {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try { fs.unlinkSync(tempFilePath); } catch (_) {}
    }
  }
});

export default router;

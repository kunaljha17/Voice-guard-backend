import { Router } from 'express';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { randomUUID } from 'crypto';
import multer from 'multer';

const router = Router();

const uploadDir = path.join(os.tmpdir(), 'voiceguard_uploads');
const tempAudioDir = path.join(uploadDir, 'persistent_temp_audio');
if (!fs.existsSync(tempAudioDir)) {
  fs.mkdirSync(tempAudioDir, { recursive: true });
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

// Temporary audio registry
const tempAudioRegistry = new Map();

// Cleanup audio older than 2 hours every 15 minutes
setInterval(() => {
  const now = Date.now();
  const maxAge = 2 * 60 * 60 * 1000;
  for (const [id, item] of tempAudioRegistry.entries()) {
    if (now - item.createdAt > maxAge) {
      if (fs.existsSync(item.filePath)) {
        try { fs.unlinkSync(item.filePath); } catch (_) {}
      }
      tempAudioRegistry.delete(id);
    }
  }
}, 15 * 60 * 1000);

/**
 * POST /api/temp-audio/upload
 */
router.post('/upload', multerWrapper, async (req, res) => {
  try {
    let originalName = 'audio_track.wav';
    let mimeType = 'audio/wav';
    let fileSize = 0;

    if (req.file) {
      originalName = req.file.originalname || originalName;
      mimeType = req.file.mimetype || 'audio/wav';
      fileSize = req.file.size;

      const ext = path.extname(originalName) || '.wav';
      const uniqueId = randomUUID();
      const savedFilePath = path.join(tempAudioDir, `temp_${uniqueId}${ext}`);

      fs.copyFileSync(req.file.path, savedFilePath);
      try { fs.unlinkSync(req.file.path); } catch (_) {}

      tempAudioRegistry.set(uniqueId, {
        id: uniqueId,
        filePath: savedFilePath,
        originalName,
        mimeType,
        size: fileSize,
        createdAt: Date.now(),
      });

      return res.json({
        success: true,
        audioId: uniqueId,
        audioUrl: `/api/temp-audio/${uniqueId}`,
        filename: originalName,
        size: fileSize,
        mimeType,
      });
    } else if (req.body?.audioData) {
      const raw = req.body.audioData;
      const cleanBase64 = raw.includes('base64,') ? raw.split('base64,')[1] : raw;
      const audioBuffer = Buffer.from(cleanBase64, 'base64');
      originalName = req.body.filename || originalName;
      mimeType = req.body.mimeType || 'audio/wav';
      fileSize = audioBuffer.length;

      const ext = path.extname(originalName) || '.wav';
      const uniqueId = randomUUID();
      const savedFilePath = path.join(tempAudioDir, `temp_${uniqueId}${ext}`);
      fs.writeFileSync(savedFilePath, audioBuffer);

      tempAudioRegistry.set(uniqueId, {
        id: uniqueId,
        filePath: savedFilePath,
        originalName,
        mimeType,
        size: fileSize,
        createdAt: Date.now(),
      });

      return res.json({
        success: true,
        audioId: uniqueId,
        audioUrl: `/api/temp-audio/${uniqueId}`,
        filename: originalName,
        size: fileSize,
        mimeType,
      });
    } else {
      return res.status(400).json({ error: 'No audio data provided to upload.' });
    }
  } catch (err) {
    console.error('[Upload Temp Error]:', err);
    return res.status(500).json({ error: err.message || 'Failed to save temporary audio.' });
  }
});

/**
 * GET /api/temp-audio/:id
 * Streams audio with HTTP Range support
 */
router.get('/:id', (req, res) => {
  const audioId = req.params.id;
  const item = tempAudioRegistry.get(audioId);

  let filePath = item ? item.filePath : null;
  let mimeType = item ? item.mimeType : 'audio/wav';

  if (!filePath || !fs.existsSync(filePath)) {
    const files = fs.readdirSync(tempAudioDir);
    const matched = files.find((f) => f.includes(audioId));
    if (matched) {
      filePath = path.join(tempAudioDir, matched);
      const ext = path.extname(matched).toLowerCase();
      const mimeMap = { '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.mp4': 'audio/mp4', '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.webm': 'audio/webm' };
      mimeType = mimeMap[ext] || 'audio/wav';
    }
  }

  if (!filePath || !fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Temporary audio not found or expired.' });
  }

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    const chunksize = end - start + 1;
    const fileStream = fs.createReadStream(filePath, { start, end });

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': mimeType,
    });
    fileStream.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': mimeType,
      'Accept-Ranges': 'bytes',
    });
    fs.createReadStream(filePath).pipe(res);
  }
});

/**
 * DELETE /api/temp-audio/:id
 */
router.delete('/:id', (req, res) => {
  const audioId = req.params.id;
  const item = tempAudioRegistry.get(audioId);
  if (item && fs.existsSync(item.filePath)) {
    try { fs.unlinkSync(item.filePath); } catch (_) {}
    tempAudioRegistry.delete(audioId);
  }
  return res.json({ success: true });
});

export default router;

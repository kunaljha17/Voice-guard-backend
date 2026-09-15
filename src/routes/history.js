import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { Scan } from '../models/Scan.js';

const router = Router();

/**
 * GET /api/history
 * Returns authenticated user's scan history from MongoDB
 */
router.get('/', requireAuth, async (req, res) => {
  try {
    const scans = await Scan.find({ userId: req.userId })
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();

    const formatted = scans.map((s) => ({
      id: s._id,
      filename: s.filename,
      duration: s.duration,
      label: s.label,
      confidence: s.confidence,
      confidencePercent: s.confidencePercent,
      spoofProb: s.spoofProb,
      isSynthetic: s.isSynthetic,
      classification: s.classification,
      anomalyTag: s.anomalyTag,
      model: s.model,
      acousticFindings: s.acousticFindings,
      glottalPulseWindow: s.glottalPulseWindow,
      spectralDiscontinuityWindow: s.spectralDiscontinuityWindow,
      neuralConsistency: s.neuralConsistency,
      harmonicDiffusionMatch: s.harmonicDiffusionMatch,
      phaseCutoff: s.phaseCutoff,
      breathAnomalySeverity: s.breathAnomalySeverity,
      rawPrediction: s.rawPrediction,
      audioUrl: s.audioUrl,
      timeAgo: getTimeAgo(s.createdAt),
      timestamp: s.createdAt,
    }));

    return res.json(formatted);
  } catch (err) {
    console.error('[History] Error:', err);
    return res.status(500).json({ error: 'Failed to fetch scan history.' });
  }
});

/**
 * DELETE /api/history/:id
 */
router.delete('/:id', requireAuth, async (req, res) => {
  try {
    const result = await Scan.findOneAndDelete({
      _id: req.params.id,
      userId: req.userId,
    });
    if (!result) {
      return res.status(404).json({ error: 'Scan not found.' });
    }
    return res.json({ success: true });
  } catch (err) {
    console.error('[History] Delete error:', err);
    return res.status(500).json({ error: 'Failed to delete scan.' });
  }
});

/**
 * GET /api/history/transcripts
 */
router.get('/transcripts', requireAuth, async (req, res) => {
  try {
    const { Transcript } = await import('../models/Transcript.js');
    const items = await Transcript.find({ userId: req.userId })
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();

    const formatted = items.map((t) => ({
      id: t._id,
      text: t.text,
      model: t.model,
      duration: t.duration,
      wordCount: t.wordCount,
      language: t.language,
      timestamp: new Date(t.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }));
    return res.json(formatted);
  } catch (err) {
    console.error('[History] Transcripts error:', err);
    return res.status(500).json({ error: 'Failed to fetch transcripts.' });
  }
});

/**
 * POST /api/history/transcripts
 */
router.post('/transcripts', requireAuth, async (req, res) => {
  try {
    const { Transcript } = await import('../models/Transcript.js');
    const { text, model, duration, wordCount, language } = req.body;
    const doc = await Transcript.create({
      userId: req.userId,
      text,
      model,
      duration,
      wordCount,
      language,
    });
    return res.status(201).json({ success: true, id: doc._id });
  } catch (err) {
    console.error('[History] Save transcript error:', err);
    return res.status(500).json({ error: 'Failed to save transcript.' });
  }
});

function getTimeAgo(date) {
  const seconds = Math.floor((Date.now() - new Date(date).getTime()) / 1000);
  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(date).toLocaleDateString();
}

export default router;

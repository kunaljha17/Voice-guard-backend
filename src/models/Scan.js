import mongoose from 'mongoose';

const scanSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    filename: { type: String, required: true },
    duration: { type: String, default: '0:00' },
    isSynthetic: { type: Boolean, default: false },
    label: { type: String, enum: ['real', 'cloned'], default: 'real' },
    confidence: { type: Number, default: 0 },
    confidencePercent: { type: Number, default: 0 },
    spoofProb: { type: Number, default: 0 },
    classification: { type: String, default: '' },
    anomalyTag: { type: String, default: '' },
    model: { type: String, default: '' },
    acousticFindings: { type: String, default: '' },
    glottalPulseWindow: { type: String, default: 'N/A' },
    spectralDiscontinuityWindow: { type: String, default: 'N/A' },
    neuralConsistency: { type: Number, default: 0 },
    harmonicDiffusionMatch: { type: Number, default: 0 },
    phaseCutoff: { type: String, default: '' },
    breathAnomalySeverity: { type: String, default: 'NORMAL' },
    rawPrediction: { type: mongoose.Schema.Types.Mixed, default: {} },
    audioUrl: { type: String, default: null },
  },
  { timestamps: true }
);

scanSchema.index({ userId: 1, createdAt: -1 });

export const Scan = mongoose.model('Scan', scanSchema);

import mongoose from 'mongoose';

const transcriptSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    text: { type: String, default: '' },
    model: { type: String, default: 'gemini-3.5-transcribe' },
    duration: { type: String, default: '0:00' },
    wordCount: { type: Number, default: 0 },
    language: { type: String, default: 'auto-detected' },
  },
  { timestamps: true }
);

transcriptSchema.index({ userId: 1, createdAt: -1 });

export const Transcript = mongoose.model('Transcript', transcriptSchema);

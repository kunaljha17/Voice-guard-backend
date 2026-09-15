import { Router } from 'express';
import { GoogleGenAI } from '@google/genai';
import { GEMINI_API_KEY } from '../config.js';

const router = Router();

let geminiClient = null;

function getGeminiClient() {
  if (!geminiClient) {
    if (!GEMINI_API_KEY) {
      throw new Error('GEMINI_API_KEY environment variable is required.');
    }
    geminiClient = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
  }
  return geminiClient;
}

/**
 * POST /api/transcribe
 * Uses Gemini models for high-fidelity audio transcription
 */
router.post('/', async (req, res) => {
  try {
    const { audioData, mimeType = 'audio/webm', prompt } = req.body;

    if (!audioData) {
      return res.status(400).json({ error: 'Missing audioData payload' });
    }

    const ai = getGeminiClient();

    const cleanBase64 = audioData.includes('base64,')
      ? audioData.split('base64,')[1]
      : audioData;

    const cleanMime = (mimeType || 'audio/webm').split(';')[0].trim() || 'audio/webm';

    const audioPart = {
      inlineData: {
        mimeType: cleanMime,
        data: cleanBase64,
      },
    };

    const instructionText =
      prompt ||
      'Transcribe this audio recording accurately verbatim with natural capitalization and punctuation. If speech is present, output only the transcribed words. If silent, output nothing.';

    const candidateModels = [
      'gemini-3.5-transcribe',
      'gemini-3.5-flash-lite',
      'gemini-3.6-flash',
      'gemini-flash-lite-latest',
      'gemini-flash-latest',
      'gemini-3.8-flash',
    ];
    let lastError = null;
    let transcriptionText = '';
    let usedModel = candidateModels[0];

    for (const modelName of candidateModels) {
      try {
        const response = await ai.models.generateContent({
          model: modelName,
          contents: [audioPart, instructionText],
        });

        const candidate = response.candidates?.[0];
        const parts = candidate?.content?.parts || [];

        for (const part of parts) {
          if (part.audioTranscription?.text) {
            transcriptionText += part.audioTranscription.text + ' ';
          } else if (part.text) {
            transcriptionText += part.text + ' ';
          }
        }

        transcriptionText = transcriptionText.trim();
        if (!transcriptionText && response.text) {
          transcriptionText = response.text.trim();
        }

        if (transcriptionText) {
          usedModel = modelName;
          lastError = null;
          console.log(`✅ [Transcribe] Successfully transcribed using ${modelName}`);
          break;
        }
      } catch (err) {
        console.warn(`[Transcription] Model ${modelName} notice:`, err?.message?.slice(0, 150) || err);
        lastError = err;
      }
    }

    if (lastError && !transcriptionText) {
      const errorMsg = lastError?.message || 'Transcription service error';
      return res.status(500).json({
        error: errorMsg.includes('INVALID_ARGUMENT')
          ? 'The audio format could not be processed. Please record again.'
          : errorMsg,
      });
    }

    return res.json({
      success: true,
      text: transcriptionText,
      model: usedModel,
    });
  } catch (error) {
    console.error('Audio transcription error:', error);
    return res.status(500).json({
      error: error?.message || 'Failed to transcribe audio',
    });
  }
});

export default router;

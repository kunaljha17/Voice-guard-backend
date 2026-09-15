import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';

dotenv.config();

export const PORT = parseInt(process.env.PORT || '5000', 10);
export const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/voiceguard';
export const JWT_SECRET = process.env.JWT_SECRET || 'voiceguard_dev_secret_change_me';
export const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';
export const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Model path resolution: self-contained within backend
const internalBackendModel = path.resolve(__dirname, '../model');
const cwdBackendModel = path.resolve(process.cwd(), 'model');
const legacyParentModel = path.resolve(process.cwd(), '../model');

export const MODEL_ROOT = (() => {
  if (fs.existsSync(internalBackendModel)) return internalBackendModel;
  if (fs.existsSync(cwdBackendModel)) return cwdBackendModel;
  if (fs.existsSync(legacyParentModel)) return legacyParentModel;
  return internalBackendModel;
})();

const defaultPtPath = path.resolve(MODEL_ROOT, 'saved_models/ssl_spoof.pt');
const defaultOnnxPath = path.resolve(MODEL_ROOT, 'models/voiceguard_v1.onnx');

const rawModelPath = process.env.MODEL_PATH || defaultPtPath;

export const MODEL_PATH = (() => {
  if (defaultPtPath && fs.existsSync(defaultPtPath)) return defaultPtPath;
  if (rawModelPath && fs.existsSync(rawModelPath)) return path.resolve(rawModelPath);
  const cwdResolved = path.resolve(process.cwd(), rawModelPath);
  if (fs.existsSync(cwdResolved)) return cwdResolved;
  if (defaultOnnxPath && fs.existsSync(defaultOnnxPath)) return defaultOnnxPath;
  return defaultPtPath || rawModelPath;
})();

export const MODEL_VERSION = process.env.MODEL_VERSION || '2.0.0-Wav2Vec2';
export const MODEL_SERVICE_PY = path.resolve(MODEL_ROOT, 'model_service.py');


// Deployed Remote ML Microservice URL
export const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'https://legendary-space-disco-5g7q7rpjrpq6cpp79-8000.app.github.dev').replace(/\/+$/, '');
export const INFERENCE_URL = (process.env.INFERENCE_URL || process.env.ML_SERVICE_URL || 'https://legendary-space-disco-5g7q7rpjrpq6cpp79-8000.app.github.dev').replace(/\/+$/, '');


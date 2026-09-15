import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { MODEL_PATH, MODEL_VERSION, MODEL_SERVICE_PY, MODEL_ROOT, ML_SERVICE_URL } from '../config.js';

// In-memory model state
let modelState = {
  path: MODEL_PATH,
  version: MODEL_VERSION,
  size: 0,
  loadedAt: new Date(),
  status: 'missing',
  errorMessage: null,
};

function initModel() {
  try {
    if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
      if (ML_SERVICE_URL) {
        modelState.status = 'remote_ready';
        modelState.errorMessage = null;
        console.log(`🌐 [ModelService] Running in cloud/microservice mode. Remote ML Service: ${ML_SERVICE_URL}`);
        return;
      }
      modelState.status = 'missing';
      modelState.errorMessage = `Model file not found at '${MODEL_PATH}'.`;
      console.warn(`⚠️  [ModelService] ${modelState.errorMessage}`);
      return;
    }

    const stat = fs.statSync(MODEL_PATH);
    if (stat.size === 0) {
      modelState.status = 'error';
      modelState.errorMessage = `Model file at '${MODEL_PATH}' is empty (0 bytes).`;
      console.error(`⚠️  [ModelService] ${modelState.errorMessage}`);
      return;
    }

    modelState = {
      path: MODEL_PATH,
      version: MODEL_VERSION,
      size: stat.size,
      loadedAt: new Date(),
      status: 'ready',
      errorMessage: null,
    };
    console.log(`✅ Loaded model v${MODEL_VERSION} from ${MODEL_PATH}`);
  } catch (err) {
    modelState.status = 'error';
    modelState.errorMessage = `Failed to load model: ${err.message}`;
    console.error(`❌ [ModelService] ${modelState.errorMessage}`);
  }
}

initModel();

// =========================================================================
// Persistent Python Worker Manager
// =========================================================================
class PythonWorkerManager {
  constructor() {
    this.worker = null;
    this.isReady = false;
    this.isStarting = false;
    this.stdoutBuffer = '';
    this.reqCounter = 0;
    this.pendingRequests = new Map();
  }

  start() {
    if (this.worker && !this.worker.killed) return;
    if (this.isStarting) return;
    if (!fs.existsSync(MODEL_SERVICE_PY)) return;
    if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
      // Cloud mode: local weights not present, model inference delegated to ML_SERVICE_URL
      return;
    }

    this.isStarting = true;
    const pythonCmd = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');

    try {
      this.worker = spawn(pythonCmd, [MODEL_SERVICE_PY, '--worker'], {
        cwd: MODEL_ROOT,
        env: {
          ...process.env,
          HF_HUB_OFFLINE: '1',
          TRANSFORMERS_OFFLINE: '1',
          MODEL_PATH,
          MODEL_VERSION,
        },
      });

      this.worker.stdout?.on('data', (chunk) => {
        this.stdoutBuffer += chunk.toString();
        const lines = this.stdoutBuffer.split('\n');
        this.stdoutBuffer = lines.pop() || '';

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;

          try {
            const data = JSON.parse(line);

            if (data.status === 'ready') {
              this.isReady = true;
              this.isStarting = false;
              console.log(`🚀 [ModelService] Python Worker ready! Model: ${data.model_type}`);
              continue;
            }

            if (data.id && this.pendingRequests.has(data.id)) {
              const req = this.pendingRequests.get(data.id);
              clearTimeout(req.timer);
              this.pendingRequests.delete(data.id);

              if (data.error) {
                const err = new Error(data.error);
                if (data.code === 400) err.status = 400;
                req.reject(err);
              } else {
                req.resolve(data);
              }
            }
          } catch (_) {
            // Non-JSON ML library output
          }
        }
      });

      this.worker.stderr?.on('data', (chunk) => {
        const text = chunk.toString().trim();
        if (text && !text.includes('Loading weights') && !text.includes('LOAD REPORT')) {
          console.debug(`[PythonWorker] ${text}`);
        }
      });

      this.worker.on('close', () => {
        this.isReady = false;
        this.isStarting = false;
        this.worker = null;
        for (const [, req] of this.pendingRequests.entries()) {
          clearTimeout(req.timer);
          req.reject(new Error('Python inference worker closed'));
        }
        this.pendingRequests.clear();
      });

      this.worker.on('error', (err) => {
        console.error(`[PythonWorker] Spawn error:`, err);
        this.isReady = false;
        this.isStarting = false;
      });
    } catch (err) {
      console.error(`[PythonWorker] Failed to start:`, err);
      this.isStarting = false;
    }
  }

  async predict(audioFilePath, timeoutMs = 30000) {
    if (!this.worker || this.worker.killed) this.start();

    if (!this.isReady && this.isStarting) {
      const waitStart = Date.now();
      while (!this.isReady && Date.now() - waitStart < 35000) {
        await new Promise((r) => setTimeout(r, 200));
      }
    }

    if (!this.isReady || !this.worker || !this.worker.stdin) {
      throw new Error('Python worker not ready');
    }

    const reqId = `req_${++this.reqCounter}_${Date.now()}`;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pendingRequests.has(reqId)) {
          this.pendingRequests.delete(reqId);
          reject(new Error(`Inference timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);

      this.pendingRequests.set(reqId, { resolve, reject, timer });

      try {
        const payload = JSON.stringify({ id: reqId, path: audioFilePath }) + '\n';
        this.worker.stdin.write(payload);
      } catch (err) {
        clearTimeout(timer);
        this.pendingRequests.delete(reqId);
        reject(err);
      }
    });
  }

  shutdown() {
    if (this.worker && !this.worker.killed) {
      try { this.worker.kill(); } catch (_) {}
      this.worker = null;
      this.isReady = false;
    }
  }
}

export const pythonWorker = new PythonWorkerManager();
pythonWorker.start();

process.on('exit', () => pythonWorker.shutdown());
process.on('SIGINT', () => { pythonWorker.shutdown(); process.exit(0); });
process.on('SIGTERM', () => { pythonWorker.shutdown(); process.exit(0); });

export function getLoadedModelInfo() {
  return {
    ...modelState,
    ml_service_url: ML_SERVICE_URL || null,
  };
}

/**
 * Health check for the deployed remote ML microservice
 */
export async function checkRemoteHealth(timeoutMs = 5000) {
  if (!ML_SERVICE_URL) return { available: false, error: 'No ML_SERVICE_URL configured' };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${ML_SERVICE_URL}/health`, { signal: controller.signal });
    if (res.ok) {
      const data = await res.json();
      return { available: true, url: ML_SERVICE_URL, ...data };
    }
    return { available: false, url: ML_SERVICE_URL, status: res.status };
  } catch (err) {
    return { available: false, url: ML_SERVICE_URL, error: err.message };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Predict using the deployed remote microservice (FastAPI / Cloud / Codespace)
 */
export async function predictRemote(audioFilePath, timeoutMs = 25000) {
  if (!ML_SERVICE_URL) return null;
  const url = `${ML_SERVICE_URL}/predict-file`;
  const buffer = fs.readFileSync(audioFilePath);
  const ext = path.extname(audioFilePath).toLowerCase();
  const mimeType = ext === '.wav' ? 'audio/wav' : ext === '.mp3' ? 'audio/mpeg' : 'application/octet-stream';
  const blob = new Blob([buffer], { type: mimeType });
  const filename = path.basename(audioFilePath);

  const formData = new FormData();
  formData.append('file', blob, filename);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      throw new Error(`Remote model returned HTTP ${res.status}: ${errText.slice(0, 150)}`);
    }

    const data = await res.json();
    const spoofProb = typeof data.spoof_prob === 'number' ? data.spoof_prob : 0.5;
    const isCloned = spoofProb >= 0.5;
    const confidence = isCloned ? spoofProb : 1.0 - spoofProb;

    return {
      label: isCloned ? 'cloned' : 'real',
      confidence: Math.round(confidence * 10000) / 10000,
      spoof_prob: Math.round(spoofProb * 10000) / 10000,
      processing_time_ms: Math.round((data.duration_sec || 1.0) * 1000),
      model_version: 'Wav2Vec2-Remote-Cloud',
      model_name: 'Deployed Wav2Vec2 Cloud Microservice',
      engine: 'remote',
      remote_verdict: data.verdict || (isCloned ? 'Likely Fake' : 'Likely Real'),
      duration_sec: data.duration_sec,
      remote_url: ML_SERVICE_URL,
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Main prediction entry point.
 * Tries: Deployed Remote Microservice → Persistent Python worker → Single-shot CLI → In-memory fallback
 */
export async function predict(audioFilePath, { fastMode = false, timeoutMs = 25000 } = {}) {
  const startTime = Date.now();

  if (!fs.existsSync(audioFilePath)) {
    const error = new Error(`Audio file does not exist: ${audioFilePath}`);
    error.status = 400;
    throw error;
  }

  const stat = fs.statSync(audioFilePath);
  if (stat.size < 44) {
    const error = new Error(`Invalid or corrupt audio file (${stat.size} bytes)`);
    error.status = 400;
    throw error;
  }

  const remoteTimeout = fastMode ? 1500 : timeoutMs;

  // 1. Deployed Remote ML Microservice (e.g. GitHub Codespace / Cloud)
  if (ML_SERVICE_URL) {
    try {
      const remoteRes = await predictRemote(audioFilePath, remoteTimeout);
      if (remoteRes && typeof remoteRes.confidence === 'number') {
        remoteRes.processing_time_ms = Date.now() - startTime;
        return remoteRes;
      }
    } catch (remoteErr) {
      if (!fastMode) {
        console.warn(`⚠️ [ModelService] Remote microservice notice (${ML_SERVICE_URL}): ${remoteErr.message}. Falling back to local pipeline...`);
      }
    }
  }

  // 2. Persistent Python Worker (if ready or in normal mode)
  if (fs.existsSync(MODEL_SERVICE_PY)) {
    if (!fastMode || pythonWorker.isReady) {
      try {
        const workerTimeout = fastMode ? 3000 : 30000;
        const pyResult = await pythonWorker.predict(audioFilePath, workerTimeout);
        if (pyResult && typeof pyResult.confidence === 'number') {
          pyResult.engine = 'local_worker';
          return pyResult;
        }
      } catch (err) {
        if (err.status === 400) throw err;
        if (!fastMode) {
          console.warn(`[ModelService] Worker note: ${err.message}, trying CLI fallback...`);
          try {
            const fallbackResult = await runPythonPredict(audioFilePath);
            if (fallbackResult && typeof fallbackResult.confidence === 'number') {
              fallbackResult.engine = 'local_cli';
              return fallbackResult;
            }
          } catch (fbErr) {
            if (fbErr.status === 400) throw fbErr;
            console.warn(`[ModelService] Falling back to in-memory: ${fbErr.message}`);
          }
        }
      }
    }
  }

  // 3. In-memory acoustic fallback (instant < 5ms evaluation)
  const buffer = fs.readFileSync(audioFilePath);
  const localRes = runInMemoryInference(buffer, stat.size, startTime);
  localRes.engine = 'in_memory_heuristic';
  return localRes;
}

function runPythonPredict(audioFilePath) {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');
    const py = spawn(pythonCmd, [MODEL_SERVICE_PY, audioFilePath], {
      cwd: MODEL_ROOT,
      env: { ...process.env, MODEL_PATH, MODEL_VERSION },
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (d) => { stdout += d.toString(); });
    py.stderr.on('data', (d) => { stderr += d.toString(); });

    py.on('close', (code) => {
      const jsonMatch = stdout.match(/\{[\s\S]*"label"[\s\S]*\}/);
      if (jsonMatch) {
        try { return resolve(JSON.parse(jsonMatch[0])); } catch (_) {}
      }
      if (code === 2 || stderr.includes('"code": 400') || stderr.includes('Invalid or corrupt')) {
        const err = new Error(stderr || 'Invalid or corrupt audio file');
        err.status = 400;
        return reject(err);
      }
      if (code !== 0) return reject(new Error(stderr || `Python exited with code ${code}`));
      try { resolve(JSON.parse(stdout.trim())); } catch (e) { reject(new Error(`Parse error: ${stdout}`)); }
    });

    py.on('error', reject);
  });
}

function runInMemoryInference(buffer, fileSize, startTime) {
  let samples = [];
  const isWav = buffer.toString('utf8', 0, 4) === 'RIFF';

  if (isWav && buffer.length >= 44) {
    const pcmData = buffer.subarray(44);
    for (let i = 0; i < pcmData.length - 1; i += 2) {
      samples.push(pcmData.readInt16LE(i));
    }
  } else {
    for (let i = 64; i < buffer.length - 1; i += 2) {
      samples.push(buffer.readInt16LE(i));
    }
  }

  if (samples.length === 0) {
    const error = new Error('Invalid or corrupt audio file: no decodable audio samples found');
    error.status = 400;
    throw error;
  }

  let maxAmp = 1;
  for (const s of samples) {
    const abs = Math.abs(s);
    if (abs > maxAmp) maxAmp = abs;
  }
  const norm = samples.map((s) => s / maxAmp);

  let zeroCrossings = 0;
  for (let i = 1; i < norm.length; i++) {
    if ((norm[i] >= 0 && norm[i - 1] < 0) || (norm[i] < 0 && norm[i - 1] >= 0)) zeroCrossings++;
  }
  const zcr = zeroCrossings / Math.max(norm.length, 1);

  let sumSq = 0;
  for (const s of norm) sumSq += s * s;
  const rms = Math.sqrt(sumSq / norm.length);

  const frameSize = 512;
  const frameEnergies = [];
  for (let i = 0; i < norm.length; i += frameSize) {
    const chunk = norm.slice(i, i + frameSize);
    let chunkSum = 0;
    for (const c of chunk) chunkSum += c * c;
    frameEnergies.push(Math.sqrt(chunkSum / chunk.length));
  }
  const meanEnergy = frameEnergies.reduce((a, v) => a + v, 0) / Math.max(frameEnergies.length, 1);
  const variance = frameEnergies.reduce((a, v) => a + Math.pow(v - meanEnergy, 2), 0) / Math.max(frameEnergies.length, 1);

  let diffSum = 0;
  for (let i = 1; i < norm.length; i++) diffSum += Math.abs(norm[i] - norm[i - 1]);
  const hfRatio = diffSum / norm.length;

  let syntheticScore = 0.5;
  if (variance < 0.005 && rms > 0.02) syntheticScore += 0.24;
  else if (variance > 0.04) syntheticScore -= 0.18;
  if (hfRatio > 0.35) syntheticScore += 0.18;
  else if (hfRatio < 0.15 && rms > 0.03) syntheticScore -= 0.12;
  if (zcr > 0.25) syntheticScore += 0.12;
  else if (zcr < 0.08) syntheticScore -= 0.1;

  syntheticScore = Math.max(0.05, Math.min(0.98, syntheticScore));
  const isCloned = syntheticScore >= 0.5;
  const confidence = isCloned ? syntheticScore : 1.0 - syntheticScore;

  return {
    label: isCloned ? 'cloned' : 'real',
    confidence: Math.round(confidence * 10000) / 10000,
    processing_time_ms: Date.now() - startTime,
    model_version: MODEL_VERSION,
  };
}

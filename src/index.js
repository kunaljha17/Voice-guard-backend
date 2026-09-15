import http from 'http';
import express from 'express';
import cors from 'cors';
import { PORT } from './config.js';
import { connectDB } from './utils/db.js';
import { getLoadedModelInfo } from './services/modelService.js';
import { setupLiveDetectWebSocket } from './services/liveDetectWs.js';

import authRoutes from './routes/auth.js';
import analyzeRoutes from './routes/analyze.js';
import historyRoutes from './routes/history.js';
import transcribeRoutes from './routes/transcribe.js';
import tempAudioRoutes from './routes/tempAudio.js';

async function startServer() {
  // Connect to MongoDB
  await connectDB();

  const app = express();

  // Middleware
  app.use(cors({ origin: true, credentials: true }));
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Health check
  app.get('/api/health', (_req, res) => {
    const modelInfo = getLoadedModelInfo();
    res.json({
      status: 'ok',
      service: 'VoiceGuard AI Backend',
      model_version: modelInfo.version,
      model_path: modelInfo.path,
      model_status: modelInfo.status,
      hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    });
  });

  // Routes
  app.use('/api/auth', authRoutes);
  app.use('/api/analyze', analyzeRoutes);
  app.use('/api/history', historyRoutes);
  app.use('/api/transcribe', transcribeRoutes);
  app.use('/api/temp-audio', tempAudioRoutes);

  // Legacy route aliases & root health check (Render compatibility)
  app.get('/', (_req, res) => res.redirect('/api/health'));
  app.get('/health', (_req, res) => res.redirect('/api/health'));


  const server = http.createServer(app);
  setupLiveDetectWebSocket(server);

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`🛡️  VoiceGuard Backend running on http://0.0.0.0:${PORT}`);
    console.log(`📡 Live Detect WebSocket listening on ws://0.0.0.0:${PORT}/live-detect`);
  });
}

startServer().catch((err) => {
  console.error('❌ Failed to start server:', err);
  process.exit(1);
});

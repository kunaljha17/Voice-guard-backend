import mongoose from 'mongoose';
import { MONGO_URI } from '../config.js';

export async function connectDB() {
  if (!MONGO_URI || MONGO_URI.includes('<db_username>') || MONGO_URI.includes('<password>')) {
    console.warn(
      '\n⚠️  ===============================================================\n' +
      '⚠️  [MongoDB Warning]: MONGO_URI in backend/.env contains placeholder "<db_username>".\n' +
      '⚠️  Please replace "<db_username>" with your actual MongoDB Atlas database username!\n' +
      '⚠️  ===============================================================\n'
    );
    return;
  }

  try {
    await mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 });
    console.log(`✅ MongoDB connected: ${mongoose.connection.host}`);
  } catch (err) {
    console.error(`❌ MongoDB connection error: ${err.message}`);
    console.warn('⚠️  Backend running, but database operations will fail until MongoDB is accessible.');
  }
}

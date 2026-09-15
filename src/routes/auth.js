import { Router } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models/User.js';
import { JWT_SECRET, JWT_EXPIRES_IN } from '../config.js';
import { requireAuth } from '../middleware/auth.js';

import mongoose from 'mongoose';

const router = Router();

function generateToken(userId) {
  return jwt.sign({ userId }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

/**
 * POST /api/auth/register
 */
router.post('/register', async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(503).json({
        error: 'Database not connected. Please replace "<db_username>" with your actual MongoDB username in backend/.env',
      });
    }

    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ error: 'Name, email, and password are required.' });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters.' });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ error: 'An account with this email already exists.' });
    }

    const user = await User.create({ name, email, password });
    const token = generateToken(user._id);

    return res.status(201).json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        createdAt: user.createdAt,
      },
    });
  } catch (err) {
    console.error('[Auth] Registration error:', err);
    if (err.code === 11000) {
      return res.status(409).json({ error: 'An account with this email already exists.' });
    }
    const message = err.name === 'MongooseServerSelectionError'
      ? 'Cannot connect to MongoDB. Please check your MONGO_URI in backend/.env'
      : (err.message || 'Registration failed. Please try again.');
    return res.status(500).json({ error: message });
  }
});

/**
 * POST /api/auth/login
 */
router.post('/login', async (req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.status(503).json({
        error: 'Database not connected. Please replace "<db_username>" with your actual MongoDB username in backend/.env',
      });
    }

    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required.' });
    }

    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const token = generateToken(user._id);

    return res.json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        createdAt: user.createdAt,
      },
    });
  } catch (err) {
    console.error('[Auth] Login error:', err);
    return res.status(500).json({ error: 'Login failed. Please try again.' });
  }
});

/**
 * GET /api/auth/me
 */
router.get('/me', requireAuth, async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found.' });
    }

    return res.json({
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        settings: user.settings,
        createdAt: user.createdAt,
      },
    });
  } catch (err) {
    console.error('[Auth] Profile fetch error:', err);
    return res.status(500).json({ error: 'Failed to fetch profile.' });
  }
});

/**
 * PUT /api/auth/settings
 */
router.put('/settings', requireAuth, async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.userId,
      { settings: req.body },
      { new: true }
    );
    if (!user) {
      return res.status(404).json({ error: 'User not found.' });
    }
    return res.json({ settings: user.settings });
  } catch (err) {
    console.error('[Auth] Settings update error:', err);
    return res.status(500).json({ error: 'Failed to update settings.' });
  }
});

export default router;

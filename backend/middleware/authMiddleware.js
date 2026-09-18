import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import ActivityLog from '../models/ActivityLog.js';

const recordActivity = (userId, req) => {
  const date = req.headers['x-client-date'] || new Date().toISOString().slice(0, 10);
  ActivityLog.updateOne(
    { user: userId, date },
    { $setOnInsert: { user: userId, date } },
    { upsert: true }
  ).catch((error) => console.error('Failed to record activity:', error.message));
};

const protect = async (req, res, next) => {
  let token;

  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith('Bearer')
  ) {
    try {
      token = req.headers.authorization.split(' ')[1];
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.user = await User.findById(decoded.id).select('-password');
      recordActivity(decoded.id, req);
      next();
    } catch (error) {
      console.error(error);
      res.status(401).json({ success: false, message: 'Not authorized, token failed' });
    }
  }

  if (!token) {
    res.status(401).json({ success: false, message: 'Not authorized, no token' });
  }
};

export { protect };

import express from 'express';
const router = express.Router();
import { getActivity } from '../controllers/activityController.js';
import { protect } from '../middleware/authMiddleware.js';

router.get('/', protect, getActivity);

export default router;

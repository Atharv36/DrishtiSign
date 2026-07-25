import express from 'express';
const router = express.Router();
import { registerUser, authUser, googleLogin } from '../controllers/authController.js';

router.post('/register', registerUser);
router.post('/login', authUser);
router.post('/google', googleLogin);

export default router;

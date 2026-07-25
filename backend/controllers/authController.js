import User from '../models/User.js';
import generateToken from '../utils/generateToken.js';
import { verifyGoogleToken } from '../services/oauthService.js';

const registerUser = async (req, res) => {
  const { name, email, password } = req.body;

  try {
    const userExists = await User.findOne({ email });

    if (userExists) {
      return res.status(400).json({ success: false, message: 'User already exists' });
    }

    const user = await User.create({
      name,
      email,
      password,
    });

    if (user) {
      res.status(201).json({
        success: true,
        data: {
          _id: user._id,
          name: user.name,
          email: user.email,
          token: generateToken(user._id),
        }
      });
    } else {
      res.status(400).json({ success: false, message: 'Invalid user data' });
    }
  } catch (error) {
    res.status(500).json({ success: false, message: 'Server Error', error: error.message });
  }
};

const authUser = async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });

    if (user && (await user.matchPassword(password))) {
      res.json({
        success: true,
        data: {
          _id: user._id,
          name: user.name,
          email: user.email,
          token: generateToken(user._id),
        }
      });
    } else {
      res.status(401).json({ success: false, message: 'Invalid email or password' });
    }
  } catch (error) {
    res.status(500).json({ success: false, message: 'Server Error', error: error.message });
  }
};

const googleLogin = async (req, res) => {
  const { credential } = req.body;
  try {
    const payload = await verifyGoogleToken(credential);
    if (!payload) {
      return res.status(400).json({ success: false, message: 'Invalid Google Token' });
    }

    const { email, name, sub: googleId } = payload;
    let user = await User.findOne({ email });

    if (!user) {
      user = await User.create({
        name,
        email,
        googleId,
        isOAuth: true,
      });
    }

    res.json({
      success: true,
      data: {
        _id: user._id,
        name: user.name,
        email: user.email,
        token: generateToken(user._id),
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, message: 'Server Error', error: error.message });
  }
};

export { registerUser, authUser, googleLogin };

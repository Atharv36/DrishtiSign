import express from 'express';
import cors from 'cors';
import authRoutes from './routes/authRoutes.js';
import activityRoutes from './routes/activityRoutes.js';

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api/auth', authRoutes);
app.use('/api/activity', activityRoutes);

app.get('/', (req, res) => {
  res.send('DrishtiSign API is running...');
});

// Error handling middleware can go here

export default app;

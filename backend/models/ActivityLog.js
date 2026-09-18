import mongoose from 'mongoose';

const activityLogSchema = mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    // Stored as the user's local calendar day, e.g. '2026-07-25'
    date: {
      type: String,
      required: true,
    },
  },
  {
    timestamps: true,
  }
);

activityLogSchema.index({ user: 1, date: 1 }, { unique: true });

const ActivityLog = mongoose.model('ActivityLog', activityLogSchema);
export default ActivityLog;

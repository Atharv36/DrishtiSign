import ActivityLog from '../models/ActivityLog.js';

const getActivity = async (req, res) => {
  try {
    const logs = await ActivityLog.find({ user: req.user._id })
      .select('date -_id')
      .lean();

    res.json({ success: true, data: logs.map((log) => log.date) });
  } catch (error) {
    res.status(500).json({ success: false, message: 'Server Error', error: error.message });
  }
};

export { getActivity };

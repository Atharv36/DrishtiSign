import dotenv from "dotenv";
import app from "./app.js";
import connectDB from "./config/db.js";

dotenv.config();

const PORT = process.env.SERVER_PORT || 5001;

// Connect DB, then start server
connectDB()
  .then(() => {
    console.log(" Database connected");

    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  })
  .catch((error) => {
    console.error(" Database connection failed:", error.message);
    process.exit(1);
  });

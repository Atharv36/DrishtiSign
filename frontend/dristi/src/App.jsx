import { useState } from "react";
import Dashboard from "./components/Dashboard";
import Practice from "./components/Practice";
import Lessons from "./components/Lessons";
import CameraModal from "./components/CameraModal";

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [showCamera, setShowCamera] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="flex gap-4 p-4 bg-white shadow">
        <button onClick={() => setPage("dashboard")}>Dashboard</button>
        <button onClick={() => setPage("practice")}>Practice</button>
        <button onClick={() => setPage("lessons")}>Lessons</button>
      </nav>

      {page === "dashboard" && (
        <Dashboard openCamera={() => setShowCamera(true)} />
      )}

      {page === "practice" && (
        <Practice openCamera={() => setShowCamera(true)} />
      )}

      {page === "lessons" && (
        <Lessons openCamera={() => setShowCamera(true)} />
      )}

      {showCamera && <CameraModal close={() => setShowCamera(false)} />}
    </div>
  );
}
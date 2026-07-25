import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background text-text-color font-sans transition-colors duration-300">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
      </div>
      
      <Toaster 
        position="bottom-right"
        toastOptions={{
          className: 'dark:bg-black/80 dark:text-white',
          style: {
            background: 'var(--bg-color)',
            color: 'var(--text-color)',
            border: '1px solid var(--accent-color)'
          }
        }} 
      />
    </Router>
  );
}

export default App;

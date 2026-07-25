import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import api from '../services/api';

const Register = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { data } = await api.post('/auth/register', { name, email, password });
      localStorage.setItem('userInfo', JSON.stringify(data.data));
      toast.success('Registration Successful!');
      navigate('/dashboard');
    } catch (error) {
      toast.error(error.response?.data?.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-[calc(100vh-73px)] px-4">
      <div className="w-full max-w-md p-8 rounded-lg shadow-xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
        <h2 className="text-3xl font-bold text-center mb-8">Create an Account</h2>
        <form onSubmit={handleRegister} className="flex flex-col gap-5">
          <div>
            <label className="block mb-2 font-medium">Name</label>
            <input 
              type="text" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-2 rounded bg-transparent border border-black/20 dark:border-white/20 focus:outline-none focus:border-accent"
              required 
            />
          </div>
          <div>
            <label className="block mb-2 font-medium">Email</label>
            <input 
              type="email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 rounded bg-transparent border border-black/20 dark:border-white/20 focus:outline-none focus:border-accent"
              required 
            />
          </div>
          <div>
            <label className="block mb-2 font-medium">Password</label>
            <input 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 rounded bg-transparent border border-black/20 dark:border-white/20 focus:outline-none focus:border-accent"
              required 
            />
          </div>
          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-3 mt-4 rounded font-semibold bg-accent text-background hover:opacity-90 transition-opacity"
          >
            {loading ? 'Processing...' : 'Register'}
          </button>
        </form>

        <p className="text-center text-sm mt-6">
          Already have an account? <Link to="/login" className="text-accent font-semibold hover:underline">Login here</Link>
        </p>
      </div>
    </div>
  );
};

export default Register;

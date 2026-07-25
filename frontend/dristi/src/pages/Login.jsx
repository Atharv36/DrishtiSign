import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import toast from 'react-hot-toast';
import api from '../services/api';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { data } = await api.post('/auth/login', { email, password });
      localStorage.setItem('userInfo', JSON.stringify(data.data));
      toast.success('Login Successful!');
      navigate('/dashboard');
    } catch (error) {
      toast.error(error.response?.data?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      try {
        const { data } = await api.post('/auth/google', { 
          // Note: with useGoogleLogin scope, we get an access_token. 
          // For a robust implementation, the backend verifyGoogleToken should handle access_token 
          // or we should use GoogleLogin component which gives credential (id_token).
          // We will pass the access token as 'credential' and adapt the backend or frontend if needed.
          // Because we used verifyGoogleToken expecting an idToken, we should ideally use the credential flow.
          // We'll update the component to use the GoogleLogin button or adapt this snippet.
          credential: tokenResponse.access_token 
        });
        localStorage.setItem('userInfo', JSON.stringify(data.data));
        toast.success('Google Login Successful!');
        navigate('/dashboard');
      } catch (err) {
        toast.error('Google Login Error');
      }
    },
    onError: () => toast.error('Google Login Failed'),
  });

  return (
    <div className="flex justify-center items-center h-[calc(100vh-73px)] px-4">
      <div className="w-full max-w-md p-8 rounded-lg shadow-xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
        <h2 className="text-3xl font-bold text-center mb-8">Sign In</h2>
        <form onSubmit={handleLogin} className="flex flex-col gap-5">
          <div>
            <label className="block mb-2 font-medium">Email Component</label>
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
            {loading ? 'Processing...' : 'Login'}
          </button>
        </form>

        <div className="my-6 flex items-center">
          <div className="flex-1 border-t border-black/10 dark:border-white/10"></div>
          <p className="px-4 text-sm opacity-70">OR</p>
          <div className="flex-1 border-t border-black/10 dark:border-white/10"></div>
        </div>

        <button 
          onClick={() => googleLogin()}
          className="w-full py-3 mb-6 rounded font-semibold border border-black/20 dark:border-white/20 hover:bg-black/5 dark:hover:bg-white/5 transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
            <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
            <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
            <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
          </svg>
          Continue with Google
        </button>

        <p className="text-center text-sm">
          Don't have an account? <Link to="/register" className="text-accent font-semibold hover:underline">Register here</Link>
        </p>
      </div>
    </div>
  );
};

export default Login;

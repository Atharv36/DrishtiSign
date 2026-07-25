import { Link, useNavigate } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';

const Navbar = () => {
  const navigate = useNavigate();
  const userInfo = JSON.parse(localStorage.getItem('userInfo'));

  const handleLogout = () => {
    localStorage.removeItem('userInfo');
    navigate('/login');
  };

  return (
    <nav className="flex items-center justify-between p-4 bg-background border-b border-black/10 dark:border-white/10">
      <Link to="/" className="text-2xl font-bold text-accent tracking-wider">
        DrishtiSign
      </Link>
      
      <div className="flex items-center gap-4">
        {userInfo ? (
          <>
            <Link to="/dashboard" className="font-medium hover:text-accent font-semibold transition-colors">
              Dashboard
            </Link>
            <button 
              onClick={handleLogout}
              className="px-4 py-2 rounded font-medium border border-accent text-accent hover:bg-accent/10 transition-colors"
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className="font-medium hover:text-accent font-semibold transition-colors">
              Login
            </Link>
            <Link 
              to="/register" 
              className="px-4 py-2 rounded font-medium bg-accent text-background hover:opacity-90 transition-opacity"
            >
              Register
            </Link>
          </>
        )}
        <ThemeToggle />
      </div>
    </nav>
  );
};

export default Navbar;

import { TypeAnimation } from 'react-type-animation';
import { Link } from 'react-router-dom';

const Landing = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-73px)] text-center px-4">
      <h1 className="text-4xl md:text-6xl font-bold mb-6 font-sans">
        Welcome to <span className="text-accent">DrishtiSign</span>
      </h1>
      
      <div className="text-xl md:text-2xl font-medium h-20 mb-10 opacity-90">
        <TypeAnimation
          sequence={[
            'Learn Sign Language',
            2000,
            'Communicate Without Barriers',
            2000,
            'AI-Powered Interpretation',
            2000,
          ]}
          wrapper="span"
          speed={50}
          repeat={Infinity}
          className="text-accent"
        />
      </div>

      <p className="max-w-2xl text-lg opacity-80 mb-12">
        An interactive platform to learn and translate sign language, bridging the communication gap using modern AI technology.
      </p>

      <div className="flex gap-6">
        <Link 
          to="/register" 
          className="px-8 py-3 rounded-md font-semibold bg-accent text-background text-lg hover:opacity-90 transition-opacity"
        >
          Get Started
        </Link>
        <Link 
          to="/login" 
          className="px-8 py-3 rounded-md font-semibold border-2 border-accent text-accent text-lg hover:bg-accent/10 transition-colors"
        >
          Login
        </Link>
      </div>
    </div>
  );
};

export default Landing;

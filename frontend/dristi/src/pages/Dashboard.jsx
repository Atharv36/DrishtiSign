import React, { useState, useEffect, Suspense, lazy } from 'react';
import { useNavigate } from 'react-router-dom';

// Lazy load the heavy ML/Camera module
const SignLearningMode = lazy(() => import('../components/SignLearningMode'));

const Dashboard = () => {
  const navigate = useNavigate();
  const [isSignModeOpen, setIsSignModeOpen] = useState(false);
  const [userName, setUserName] = useState('');

  useEffect(() => {
    const userInfoData = localStorage.getItem('userInfo');
    if (!userInfoData) {
      navigate('/login');
      return;
    }
    
    try {
        const parsed = JSON.parse(userInfoData);
        if (parsed && parsed.name) {
            setUserName(parsed.name.split(' ')[0]);
        }
    } catch(e) {
        // Fallback or ignore
    }
  }, [navigate]);

  return (
    <div className="min-h-[calc(100vh-73px)] p-6 md:p-12 transition-colors duration-300 relative overflow-hidden">
      
      {/* No Background Elements - Static Design */}

      <div className="max-w-7xl mx-auto relative z-10">
          
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 border-b border-gray-200 dark:border-gray-800 pb-6">
            <div>
                <p className="text-[var(--accent-color)] font-medium mb-2 opacity-80 tracking-widest text-sm uppercase">Welcome back</p>
                <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-[var(--text-color)]">
                  Dashboard {userName && <span className="opacity-80">, {userName}</span>}
                </h1>
            </div>
        </div>

        {/* 3 Explicit Feature Modules Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

            {/* 1. Sign Learning (Flashcard Mode) ACTIVE */}
            <div className="col-span-1 lg:col-span-3 group relative">
                <div className="bg-white dark:bg-[#0f172a] border border-[var(--accent-color)] rounded-3xl p-8 sm:p-10 shadow-xl relative overflow-hidden flex flex-col md:flex-row justify-between items-center gap-8">
                    
                    {/* Static Card Background */}

                    <div className="relative z-10 w-full md:w-2/3">
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-full text-xs font-bold uppercase tracking-wider mb-4 border border-emerald-500/20">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            Active Feature
                        </div>
                        <h2 className="text-3xl sm:text-4xl font-bold mb-4 text-[var(--text-color)]">Sign Learning <span className="opacity-50 font-normal">| Flashcard Mode</span></h2>
                        <p className="text-gray-600 dark:text-gray-400 text-lg mb-8 max-w-2xl">
                            Master sign language using our real-time AI accuracy model. Follow the 3D hand demonstrations and test your skills efficiently with zero lag.
                        </p>
                        
                        <button 
                            onClick={() => setIsSignModeOpen(true)}
                            className="bg-[var(--text-color)] text-[var(--bg-color)] px-8 py-4 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all flex items-center gap-3 transform hover:-translate-y-1 shadow-lg"
                        >
                            Open Flashcards
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                        </button>
                    </div>

                    {/* Visual Graphic Placeholder for Active module */}
                    <div className="relative z-10 w-full md:w-1/3 flex justify-center md:justify-end">
                        <div className="w-48 h-48 bg-gray-100 dark:bg-gray-800 rounded-full border-8 border-white dark:border-black shadow-2xl flex items-center justify-center overflow-hidden relative group-hover:scale-105 transition-transform duration-500">
                             <div className="absolute inset-0 bg-[var(--accent-color)] opacity-20"></div>
                             <svg className="w-20 h-20 text-[var(--accent-color)] relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                        </div>
                    </div>

                </div>
            </div>

            {/* 2. Sign → Text  COMING SOON */}
            <div className="col-span-1 lg:col-span-1">
                <div className="h-full bg-gray-50/80 dark:bg-black/60 border border-gray-200 dark:border-gray-800 rounded-3xl p-8 opacity-80 cursor-not-allowed">
                    
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-200 dark:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-full text-xs font-bold uppercase tracking-wider mb-6">
                         Coming Soon
                    </div>

                    <h3 className="text-2xl font-bold mb-3 text-[var(--text-color)]">Sign → Text</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                        Continuous gesture-to-sentence translation. Stream your signs and instantly receive natural text outputs.
                    </p>

                    <div className="space-y-4 pointer-events-none opacity-50">
                        {/* Fake Webcam Preview */}
                        <div className="w-full h-32 bg-gray-200 dark:bg-gray-800 rounded-xl flex items-center justify-center border border-gray-300 dark:border-gray-700">
                            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path></svg>
                        </div>
                        {/* Fake Text Output */}
                        <div className="w-full h-20 bg-gray-100 dark:bg-gray-900 rounded-xl p-3 border border-gray-200 dark:border-gray-800 flex items-start gap-2">
                             <div className="w-2 h-4 bg-gray-300 dark:bg-gray-700 animate-pulse mt-1"></div>
                        </div>
                    </div>

                </div>
            </div>

            {/* 3. Text → Sign  COMING SOON */}
            <div className="col-span-1 lg:col-span-2">
                 <div className="h-full bg-gray-50/80 dark:bg-black/60 border border-gray-200 dark:border-gray-800 rounded-3xl p-8 opacity-80 cursor-not-allowed">
                     
                     <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-200 dark:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-full text-xs font-bold uppercase tracking-wider mb-6">
                         Coming Soon
                    </div>

                    <h3 className="text-2xl font-bold mb-3 text-[var(--text-color)]">Text → Sign</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                        Type out sentences or import documents to watch our 3D Avatar sequentially sign your text.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pointer-events-none opacity-50">
                         {/* Disabled Input */}
                         <div className="w-full">
                            <label className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-1 block">Input Text</label>
                            <div className="w-full h-32 bg-white dark:bg-[#0f172a] rounded-xl border border-gray-300 dark:border-gray-700 p-4 shadow-inner">
                                <span className="text-gray-400">Type something...</span>
                            </div>
                         </div>

                         {/* Sequential Avatar output skeleton */}
                         <div className="w-full flex gap-2">
                             <div className="flex-1 bg-gray-200 dark:bg-gray-800 rounded-xl h-32 flex items-end p-2 border border-gray-300 dark:border-gray-700 relative overflow-hidden">
                                  <div className="absolute -bottom-4 -right-4 w-16 h-16 rounded-full bg-indigo-500/20"></div>
                             </div>
                             <div className="w-12 flex items-center justify-center">
                                 <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                             </div>
                             <div className="flex-1 bg-gray-200 dark:bg-gray-800 rounded-xl h-32 flex items-end p-2 border border-gray-300 dark:border-gray-700 relative overflow-hidden">
                                  <div className="absolute -bottom-4 -left-4 w-16 h-16 rounded-full bg-teal-500/20"></div>
                             </div>
                         </div>
                    </div>

                 </div>
            </div>

        </div>

      </div>

      {/* Lazy Load Sign Learning Feature */}
      {isSignModeOpen && (
        <Suspense fallback={
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="bg-white dark:bg-[#0f172a] p-8 rounded-3xl flex flex-col justify-center items-center">
                     <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-color)] mb-4"></div>
                     <p className="font-medium animate-pulse">Initializing Interface...</p>
                </div>
            </div>
        }>
          <SignLearningMode close={() => setIsSignModeOpen(false)} />
        </Suspense>
      )}

    </div>
  );
};

export default Dashboard;

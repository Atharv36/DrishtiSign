import React, { useState, useEffect, Suspense, lazy } from 'react';
import { useNavigate } from 'react-router-dom';
import ActivityCalendar from '../components/ActivityCalendar';

// Lazy load the heavy ML/Camera module
const SignLearningMode = lazy(() => import('../components/SignLearningMode'));
const SignPracticeMode = lazy(() => import('../components/SignPracticeMode'));
const SignToText = lazy(() => import('../components/SignToText'));
const TextToSign = lazy(() => import('../components/TextToSign'));

const Dashboard = () => {
  const navigate = useNavigate();
  const [isSignModeOpen, setIsSignModeOpen] = useState(false);
  const [isPracticeModeOpen, setIsPracticeModeOpen] = useState(false);
  const [isSignToTextOpen, setIsSignToTextOpen] = useState(false);
  const [isTextToSignOpen, setIsTextToSignOpen] = useState(false);
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">{/* 1. Sign Learning (Flashcard Mode) ACTIVE */}
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
                        
                        <div className="flex flex-wrap gap-3">
                            <button
                                onClick={() => setIsSignModeOpen(true)}
                                className="bg-[var(--text-color)] text-[var(--bg-color)] px-8 py-4 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all flex items-center gap-3 transform hover:-translate-y-1 shadow-lg"
                            >
                                Start Learning
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                            </button>
                            <button
                                onClick={() => setIsPracticeModeOpen(true)}
                                className="bg-transparent border-2 border-[var(--text-color)] text-[var(--text-color)] px-8 py-4 rounded-xl font-bold tracking-wide hover:bg-[var(--text-color)] hover:text-[var(--bg-color)] transition-all flex items-center gap-3 transform hover:-translate-y-1"
                            >
                                Start Practice
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                            </button>
                        </div>
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

            {/* 2. Sign → Text  ACTIVE */}
            <div className="col-span-1 lg:col-span-1">
                <div className="h-full bg-white dark:bg-[#0f172a] border border-indigo-400/40 rounded-3xl p-8 shadow-xl flex flex-col">

                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-500/10 text-indigo-500 dark:text-indigo-300 rounded-full text-xs font-bold uppercase tracking-wider mb-6 border border-indigo-500/20 w-fit">
                        <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
                        Active Feature
                    </div>

                    <h3 className="text-2xl font-bold mb-3 text-[var(--text-color)]">Sign → Text</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                        Continuous gesture-to-sentence translation. Hold a letter to spell it, use Space/Backspace to punctuate.
                    </p>

                    <div className="mt-auto">
                        <button
                            onClick={() => setIsSignToTextOpen(true)}
                            className="w-full bg-[var(--text-color)] text-[var(--bg-color)] px-6 py-3.5 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all flex items-center justify-center gap-2"
                        >
                            Start Translating
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                        </button>
                    </div>

                </div>
            </div>

            {/* 3. Text → Sign  ACTIVE */}
            <div className="col-span-1 lg:col-span-2">
                 <div className="h-full bg-white dark:bg-[#0f172a] border border-teal-400/40 rounded-3xl p-8 shadow-xl flex flex-col md:flex-row justify-between items-center gap-8">

                    <div className="w-full md:w-2/3">
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-teal-500/10 text-teal-500 dark:text-teal-300 rounded-full text-xs font-bold uppercase tracking-wider mb-4 border border-teal-500/20">
                            <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span>
                            Active Feature
                        </div>

                        <h3 className="text-3xl font-bold mb-3 text-[var(--text-color)]">Text → Sign</h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 max-w-lg">
                            Type a sentence and our 3D avatar signs it back — using whole-word signs where it knows them, fingerspelling everything else.
                        </p>

                        <button
                            onClick={() => setIsTextToSignOpen(true)}
                            className="bg-[var(--text-color)] text-[var(--bg-color)] px-6 py-3.5 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all flex items-center gap-2 w-fit"
                        >
                            Start Translating
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                        </button>
                    </div>

                    <div className="w-full md:w-1/3 flex justify-center md:justify-end">
                        <div className="w-40 h-40 bg-gray-100 dark:bg-gray-800 rounded-full border-8 border-white dark:border-black shadow-2xl flex items-center justify-center overflow-hidden relative">
                             <div className="absolute inset-0 bg-teal-400 opacity-20"></div>
                             <svg className="w-16 h-16 text-teal-400 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                        </div>
                    </div>

                 </div>
            </div>

            {/* Activity Calendar */}
            <div className="col-span-1 lg:col-span-3">
                <h2 className="text-2xl font-bold mb-4 text-[var(--text-color)]">Your Activity</h2>
                <div className="max-w-md">
                    <ActivityCalendar />
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

      {/* Lazy Load Sign Practice Feature */}
      {isPracticeModeOpen && (
        <Suspense fallback={
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="bg-white dark:bg-[#0f172a] p-8 rounded-3xl flex flex-col justify-center items-center">
                     <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-color)] mb-4"></div>
                     <p className="font-medium animate-pulse">Initializing Interface...</p>
                </div>
            </div>
        }>
          <SignPracticeMode close={() => setIsPracticeModeOpen(false)} />
        </Suspense>
      )}

      {/* Lazy Load Sign to Text Feature */}
      {isSignToTextOpen && (
        <Suspense fallback={
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="bg-white dark:bg-[#0f172a] p-8 rounded-3xl flex flex-col justify-center items-center">
                     <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-color)] mb-4"></div>
                     <p className="font-medium animate-pulse">Initializing Interface...</p>
                </div>
            </div>
        }>
          <SignToText close={() => setIsSignToTextOpen(false)} />
        </Suspense>
      )}

      {/* Lazy Load Text to Sign Feature */}
      {isTextToSignOpen && (
        <Suspense fallback={
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="bg-white dark:bg-[#0f172a] p-8 rounded-3xl flex flex-col justify-center items-center">
                     <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-color)] mb-4"></div>
                     <p className="font-medium animate-pulse">Initializing Interface...</p>
                </div>
            </div>
        }>
          <TextToSign close={() => setIsTextToSignOpen(false)} />
        </Suspense>
      )}

    </div>
  );
};

export default Dashboard;

import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { SIGNS as FALLBACK_SIGNS, fetchSigns, fetchWordSigns } from '../constants/signs';

const SUCCESS_POPUP_MS = 5000;
const ROUND_TIME_MS = 20000; // Hidden window - no visible countdown by design.

// See SignLearningMode: the server already gates on confidence, so we accept a
// confidently-detected matching letter at a lower bar to make passing reliable.
const MATCH_CONFIDENCE = 0.5;

const randomIndexExcluding = (list, exclude) => {
  const available = list.map((_, i) => i).filter((i) => i !== exclude);
  if (!available.length) return 0;
  return available[Math.floor(Math.random() * available.length)];
};

export default function SignPracticeMode({ close }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  // Data from Python ML
  const [processedImage, setProcessedImage] = useState(null);
  const [detectedLabel, setDetectedLabel] = useState("");
  const [accuracy, setAccuracy] = useState(0);

  // Letters and words are quizzed by two different models - a letter is a
  // handshape, a word sign is a movement - so the mode is explicit.
  const [mode, setMode] = useState('letters');
  const [letterSigns, setLetterSigns] = useState(FALLBACK_SIGNS);
  const [wordSigns, setWordSigns] = useState([]);

  const flashcards = mode === 'words' ? wordSigns : letterSigns;

  // Quiz state - randomized order, no demo image (this is recall practice, not a lesson)
  const [cardIndex, setCardIndex] = useState(() => Math.floor(Math.random() * FALLBACK_SIGNS.length));
  const targetSign = flashcards[cardIndex] || flashcards[0];
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    fetchSigns().then(setLetterSigns);
    fetchWordSigns().then(setWordSigns);
  }, []);

  const socketRef = useRef(null);
  const isProcessingRef = useRef(false);
  const modeRef = useRef(mode);   // read by the frame loop without resubscribing
  const advanceLockRef = useRef(false); // Prevent double-advancing per round
  const successTimerRef = useRef(null);
  const roundTimerRef = useRef(null);

  const resetDetectionState = () => {
    setDetectedLabel("");
    setAccuracy(0);
  };

  const clearTimers = () => {
    if (successTimerRef.current) {
      clearTimeout(successTimerRef.current);
      successTimerRef.current = null;
    }
    if (roundTimerRef.current) {
      clearTimeout(roundTimerRef.current);
      roundTimerRef.current = null;
    }
  };

  const goToNextCard = () => {
    clearTimers();
    advanceLockRef.current = false;
    setShowSuccess(false);
    resetDetectionState();
    setCardIndex((current) => randomIndexExcluding(flashcards, current));
  };

  const skipCard = () => goToNextCard();

  const switchMode = (next) => {
    if (next === mode) return;
    clearTimers();
    advanceLockRef.current = false;
    setShowSuccess(false);
    resetDetectionState();
    setMode(next);
    setCardIndex(0);
  };

  const isCorrect = detectedLabel === targetSign && accuracy > MATCH_CONFIDENCE;
  const matchPercentage = detectedLabel === targetSign ? Math.round(accuracy * 100) : 0;

  // Passed: show the success popup, then move on once it closes.
  useEffect(() => {
    if (isCorrect && !advanceLockRef.current) {
      advanceLockRef.current = true;
      if (roundTimerRef.current) {
        clearTimeout(roundTimerRef.current);
        roundTimerRef.current = null;
      }
      setShowSuccess(true);
      successTimerRef.current = setTimeout(() => {
        setShowSuccess(false);
        goToNextCard();
      }, SUCCESS_POPUP_MS);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCorrect]);

  // Hidden 20s window per sign: if the camera is running and time runs out
  // without a correct match, silently move on - no popup, no message.
  useEffect(() => {
    if (!isActive || showSuccess) return undefined;

    roundTimerRef.current = setTimeout(() => {
      if (!advanceLockRef.current) {
        advanceLockRef.current = true;
        goToNextCard();
      }
    }, ROUND_TIME_MS);

    return () => {
      if (roundTimerRef.current) {
        clearTimeout(roundTimerRef.current);
        roundTimerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cardIndex, isActive]);

  useEffect(() => { modeRef.current = mode; }, [mode]);

  useEffect(() => {
    return () => clearTimers();
  }, []);

  useEffect(() => {
    socketRef.current = io('http://localhost:5002');

    socketRef.current.on('connect', () => {
      console.log('Connected to ML Server');
    });

    socketRef.current.on('processed_frame', (data) => {
      if (typeof data === 'string') {
          setProcessedImage(data);
      } else {
          setProcessedImage(data.image);
          setDetectedLabel(data.label);
          setAccuracy(data.confidence);
      }
      isProcessingRef.current = false;
    });

    // Word mode is served by the temporal model on a different event.
    socketRef.current.on('word_processed_frame', (data) => {
      if (data.image !== undefined) setProcessedImage(data.image);
      if (data.label !== undefined) setDetectedLabel(data.label);
      if (data.confidence !== undefined) setAccuracy(data.confidence);
      isProcessingRef.current = false;
    });

    return () => {
        socketRef.current.disconnect();
        stopCamera();
    };
  }, []);

  useEffect(() => {
    let animationId;
    let lastSendTime = 0;

    const sendFrame = (timestamp) => {
      if (isActive && videoRef.current && canvasRef.current && !showSuccess) {

        if (!isProcessingRef.current || (timestamp - lastSendTime > 500)) {

            isProcessingRef.current = true;
            lastSendTime = timestamp;

            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });

            if (video.videoWidth > 0 && video.videoHeight > 0) {
                const scale = Math.min(1, 480 / video.videoHeight);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const base64Data = canvas.toDataURL('image/jpeg', 0.6);
                socketRef.current.emit(
                    modeRef.current === 'words' ? 'word_frame' : 'video_frame',
                    base64Data);
            } else {
                isProcessingRef.current = false;
            }
        }
      }
      animationId = requestAnimationFrame(sendFrame);
    };

    if (isActive) {
        animationId = requestAnimationFrame(sendFrame);
    }

    return () => {
        if (animationId) cancelAnimationFrame(animationId);
    }
  }, [isActive, showSuccess]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsActive(true);
      }
    } catch (err) {
      console.error("Error accessing camera: ", err);
      alert("Could not access camera. Please ensure permissions are granted.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setProcessedImage(null);
    resetDetectionState();
    clearTimers();
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">

        {/* Header */}
        <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-4">
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-orange-600">
                    Sign Practice
                </h2>

                {/* Letters and words are quizzed by different models. */}
                <div className="flex bg-gray-100 dark:bg-gray-800 rounded-xl p-1 gap-1">
                    {[['letters', 'Letters'], ['words', 'Words']].map(([key, text]) => (
                        <button
                            key={key}
                            onClick={() => switchMode(key)}
                            disabled={key === 'words' && wordSigns.length === 0}
                            className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                                mode === key
                                    ? 'bg-white dark:bg-[#0f172a] text-amber-600 dark:text-amber-400 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                            }`}
                        >
                            {text}
                        </button>
                    ))}
                </div>
            </div>
            <button onClick={close} className="text-gray-500 hover:text-red-500 transition-colors p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start h-[600px] relative">

          {/* Main Feed Container */}
          <div className="md:col-span-8 bg-gray-100 dark:bg-black rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 shadow-inner relative h-full flex flex-col items-center justify-center">

            <video ref={videoRef} className="hidden" muted playsInline />
            <canvas ref={canvasRef} className="hidden" />

            {!isActive ? (
                <div className="text-center p-8 flex flex-col items-center">
                    <div className="w-20 h-20 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4 shadow-sm">
                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                    </div>
                    <p className="text-gray-500 dark:text-gray-400 font-medium text-lg">Camera is inactive</p>
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Start the camera to begin the quiz. No demo photo this time — recall it yourself.</p>
                </div>
            ) : processedImage ? (
                <>
                    <img src={processedImage} alt="Live Stream" className="w-full h-full object-cover" />

                    <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
                        <div className="bg-black/70 backdrop-blur-md p-3 rounded-xl border border-white/10 text-white min-w-[200px]">
                            <p className="text-xs uppercase tracking-wider text-gray-300 mb-1">Live Tracking</p>
                            <div className="flex items-end gap-3">
                                <span className="text-3xl font-bold">{detectedLabel || "—"}</span>
                                <span className="text-emerald-400 font-mono mb-1">{Math.round(accuracy * 100)}%</span>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="flex flex-col items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-500 mb-4"></div>
                    <p className="text-gray-500 dark:text-gray-400 animate-pulse">Connecting to ML Engine...</p>
                </div>
            )}

            {isActive && (
                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full flex items-center gap-2 border border-white/10">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    <span className="text-xs text-white font-medium tracking-wide">LIVE</span>
                </div>
            )}
          </div>

          {/* Quiz Prompt Controls */}
          <div className="md:col-span-4 flex flex-col h-full space-y-4">

            <div className={`flex-1 rounded-2xl p-6 flex flex-col items-center justify-center text-center transition-colors duration-500 border shadow-sm ${isCorrect ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-500' : 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-800'}`}>
                <h3 className="text-gray-500 dark:text-gray-400 font-medium tracking-wide text-sm uppercase mb-2">Sign this</h3>

                <div className="text-8xl font-black text-gray-900 dark:text-white my-4">
                    {targetSign}
                </div>

                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full mt-6 overflow-hidden">
                    <div
                        className="h-full bg-amber-500 transition-all duration-300 ease-out"
                        style={{ width: `${matchPercentage}%` }}
                    ></div>
                </div>
                <p className="text-xs text-gray-500 mt-2">{matchPercentage}% Match</p>
            </div>

            <div className="flex gap-2">
                <button onClick={skipCard} className="flex-1 py-3 px-4 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 font-medium transition-colors text-[var(--text-color)]">
                    Next
                </button>
            </div>

            <div className="pt-2">
                {!isActive ? (
                    <button
                        onClick={startCamera}
                        className="w-full bg-[var(--text-color)] text-[var(--bg-color)] py-3.5 px-4 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all shadow-md flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Start Practice
                    </button>
                ) : (
                    <button
                        onClick={stopCamera}
                        className="w-full bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/20 hover:border-red-500 py-3.5 px-4 rounded-xl font-bold transition-all flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 10h6v4H9z"></path></svg>
                        Stop Model
                    </button>
                )}
            </div>

          </div>

          {/* Success Popup (auto-closes after 5s) */}
          {showSuccess && (
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center rounded-2xl z-10 animate-in fade-in duration-200">
                <div className="bg-white dark:bg-[#0f172a] rounded-3xl px-10 py-8 shadow-2xl flex flex-col items-center text-center border border-emerald-500/30">
                    <div className="w-16 h-16 rounded-full bg-emerald-500 flex items-center justify-center mb-4 shadow-lg">
                        <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white">You did it!</h3>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">Sign "{targetSign}" recognized successfully.</p>
                    <button
                        onClick={goToNextCard}
                        className="mt-6 bg-emerald-500 hover:bg-emerald-600 text-white px-8 py-3 rounded-xl font-bold tracking-wide transition-all flex items-center gap-2"
                    >
                        Next
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </button>
                </div>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}

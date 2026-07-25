import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';

const FLASHCARDS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", 
  "Hello", "Thank You", "I Love You", "Yes", "No", "Please", "Sorry"
];

export default function SignLearningMode({ close }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  
  // Data from Python ML
  const [processedImage, setProcessedImage] = useState(null);
  const [detectedLabel, setDetectedLabel] = useState("");
  const [accuracy, setAccuracy] = useState(0);

  // Flashcard state
  const [targetSign, setTargetSign] = useState(FLASHCARDS[0]);
  const socketRef = useRef(null);
  const isProcessingRef = useRef(false); // Add ref to prevent queue buildup

  const setRandomSign = () => {
      const available = FLASHCARDS.filter(s => s !== targetSign);
      const random = available[Math.floor(Math.random() * available.length)];
      setTargetSign(random);
  };

  // Auto-advance if accuracy is high
  useEffect(() => {
      if (detectedLabel === targetSign && accuracy > 0.70) {
          const timer = setTimeout(() => {
              setRandomSign();
          }, 600); // Super fast 600ms delay to feel extremely snappy
          return () => clearTimeout(timer);
      }
  }, [detectedLabel, accuracy, targetSign]);

  useEffect(() => {
    // Only connect when component mounts
    socketRef.current = io('http://localhost:5002');
    
    socketRef.current.on('connect', () => {
      console.log('Connected to ML Server');
    });

    socketRef.current.on('processed_frame', (data) => {
      // The new ML server emits a JSON object
      if (typeof data === 'string') {
          // Fallback if local server isn't updated string -> base64
          setProcessedImage(data);
      } else {
          setProcessedImage(data.image);
          setDetectedLabel(data.label);
          setAccuracy(data.confidence);
      }
      
      // Unlock frame sending, allowing the next camera frame to transmit!
      isProcessingRef.current = false;
    });

    return () => {
        socketRef.current.disconnect();
        stopCamera();
    };
  }, []);

  useEffect(() => {
    let animationId;
    
    // Fallback: If python server crashes/drops a frame, unlock after 1 second so it doesn't freeze permanently
    let lastSendTime = 0;
    
    const sendFrame = (timestamp) => {
      // If we are already processing a frame, OR the camera is off, don't send anything.
      // Emergency timeout: if 500ms passed and no response, force unlock (dropped packet).
      if (isActive && videoRef.current && canvasRef.current) {
        
        if (!isProcessingRef.current || (timestamp - lastSendTime > 500)) {
            
            isProcessingRef.current = true;
            lastSendTime = timestamp;
            
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                // Downscale frame to lower network payload & python processing time
                const scale = Math.min(1, 480 / video.videoHeight);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;
                
                // Draw current video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                // Convert to compressed jpeg (60% quality is fine for MediaPipe)
                const base64Data = canvas.toDataURL('image/jpeg', 0.6);
                // Send to server
                socketRef.current.emit('video_frame', base64Data);
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
  }, [isActive]);

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
    setDetectedLabel("");
    setAccuracy(0);
  };

  const nextCard = () => {
      setRandomSign();
  };

  const prevCard = () => {
      setRandomSign();
  };

  // Check if user is signing the correct active flashcard
  const isCorrect = detectedLabel === targetSign && accuracy > 0.6;
  const matchPercentage = detectedLabel === targetSign ? Math.round(accuracy * 100) : 0;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">
        
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-emerald-600">
                Sign Learning: Flashcard Mode
            </h2>
            <button onClick={close} className="text-gray-500 hover:text-red-500 transition-colors p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start h-[600px]">
          
          {/* Main Feed Container (Webcam + 3D rendering stream) */}
          <div className="md:col-span-8 bg-gray-100 dark:bg-black rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 shadow-inner relative h-full flex flex-col items-center justify-center">
            
            <video ref={videoRef} className="hidden" muted playsInline />
            <canvas ref={canvasRef} className="hidden" />

            {!isActive ? (
                <div className="text-center p-8 flex flex-col items-center">
                    <div className="w-20 h-20 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4 shadow-sm">
                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                    </div>
                    <p className="text-gray-500 dark:text-gray-400 font-medium text-lg">Camera is inactive</p>
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Start the camera to begin the flashcard challenge.</p>
                </div>
            ) : processedImage ? (
                <>
                    {/* Display processed output from python server which ALREADY contains the 3D rendered avatar */}
                    <img src={processedImage} alt="Live Stream" className="w-full h-full object-cover" />
                    
                    {/* Floating Accuracy Metric */}
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
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500 mb-4"></div>
                    <p className="text-gray-500 dark:text-gray-400 animate-pulse">Connecting to ML Engine...</p>
                </div>
            )}
            
            {/* Live Indicator */}
            {isActive && (
                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full flex items-center gap-2 border border-white/10">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    <span className="text-xs text-white font-medium tracking-wide">LIVE</span>
                </div>
            )}
          </div>

          {/* Flashcard Target Controls */}
          <div className="md:col-span-4 flex flex-col h-full space-y-4">
            
            <div className={`flex-1 rounded-2xl p-6 flex flex-col items-center justify-center text-center transition-colors duration-500 border shadow-sm ${isCorrect ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-500' : 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-800'}`}>
                <h3 className="text-gray-500 dark:text-gray-400 font-medium tracking-wide text-sm uppercase mb-2">Target Sign</h3>
                
                <div className="text-8xl font-black text-gray-900 dark:text-white my-4 relative">
                    {targetSign}
                    {isCorrect && (
                        <div className="absolute -top-4 -right-8 bg-emerald-500 text-white rounded-full p-1 animate-bounce shadow-lg border-2 border-white dark:border-gray-900">
                             <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                        </div>
                    )}
                </div>

                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full mt-6 overflow-hidden">
                    <div 
                        className="h-full bg-emerald-500 transition-all duration-300 ease-out" 
                        style={{ width: `${matchPercentage}%` }}
                    ></div>
                </div>
                <p className="text-xs text-gray-500 mt-2">{matchPercentage}% Match</p>
            </div>
            
            {/* Nav Controls */}
            <div className="flex gap-2">
                <button onClick={prevCard} className="flex-1 py-3 px-4 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 font-medium transition-colors text-[var(--text-color)]">
                    Previous
                </button>
                <button onClick={nextCard} className="flex-1 py-3 px-4 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 font-medium transition-colors text-[var(--text-color)]">
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
                        Start Learning
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
        </div>

      </div>
    </div>
  );
}

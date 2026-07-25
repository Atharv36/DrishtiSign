import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';

export default function CameraModal({ close }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // Connect to Python ML Server
    socketRef.current = io('http://localhost:5002');
    
    socketRef.current.on('connect', () => {
      console.log('Connected to ML Server');
    });

    socketRef.current.on('processed_frame', (data) => {
      setProcessedImage(data);
    });

    return () => {
        socketRef.current.disconnect();
        stopCamera();
    };
  }, []);

  useEffect(() => {
    let animationId;
    
    const sendFrame = () => {
      if (isActive && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Draw current video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            // Convert to base64 jpeg
            const base64Data = canvas.toDataURL('image/jpeg', 0.8);
            // Send to server
            socketRef.current.emit('video_frame', base64Data);
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
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">
        
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-[var(--accent-color)] to-emerald-500">
                Live Sign Translator
            </h2>
            <button onClick={close} className="text-gray-500 hover:text-red-500 transition-colors p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
          
          {/* Main Feed Container */}
          <div className="md:col-span-8 bg-gray-100 dark:bg-black rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 shadow-inner relative min-h-[400px] flex items-center justify-center">
            
            {/* Hidden raw video element */}
            <video ref={videoRef} className="hidden" muted playsInline />
            <canvas ref={canvasRef} className="hidden" />

            {!isActive ? (
                <div className="text-center p-8 flex flex-col items-center">
                    <div className="w-20 h-20 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4 shadow-sm">
                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                    </div>
                    <p className="text-gray-500 dark:text-gray-400 font-medium text-lg">Camera is inactive</p>
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Start the camera to begin real-time sign language translation and 3D avatar mirroring.</p>
                </div>
            ) : processedImage ? (
                // Display processed output from python server
                <img src={processedImage} alt="Live Stream" className="w-full h-auto object-contain max-h-[600px]" />
            ) : (
                <div className="flex flex-col items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-color)] mb-4"></div>
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

          {/* Controls & Instructions */}
          <div className="md:col-span-4 flex flex-col h-full space-y-6">
            
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-5 border border-gray-200 dark:border-gray-800 h-full flex flex-col justify-between shadow-sm">
                
                <div>
                    <h3 className="font-semibold text-lg mb-2 text-gray-800 dark:text-gray-200">How to use</h3>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-3 mb-6">
                        <li className="flex items-start gap-2">
                            <span className="text-[var(--accent-color)] mt-0.5">•</span>
                            Ensure your hands are clearly visible in the frame.
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-[var(--accent-color)] mt-0.5">•</span>
                            A well-lit environment improves detection accuracy.
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-[var(--accent-color)] mt-0.5">•</span>
                            Follow the 3D avatar to learn standard poses.
                        </li>
                    </ul>
                </div>

                <div className="space-y-3 mt-auto">
                    {!isActive ? (
                        <button 
                            onClick={startCamera} 
                            className="w-full bg-[var(--text-color)] text-[var(--bg-color)] py-3.5 px-4 rounded-xl font-bold tracking-wide hover:bg-[var(--accent-color)] transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-md flex items-center justify-center gap-2"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            Start Translation
                        </button>
                    ) : (
                        <button 
                            onClick={stopCamera} 
                            className="w-full bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/20 hover:border-red-500 py-3.5 px-4 rounded-xl font-bold transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 10h6v4H9z"></path></svg>
                            Stop Connection
                        </button>
                    )}
                </div>
            </div>
            
          </div>
        </div>

      </div>
    </div>
  );
}
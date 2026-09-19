import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';

export default function SignToText({ close }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  // 'letters' uses the single-frame letter model (fingerspelling);
  // 'words' uses the temporal word model. They're genuinely different models -
  // a letter is a handshape, a word sign is a movement - so the mode is chosen
  // explicitly rather than guessed at.
  const [mode, setMode] = useState('letters');

  const [processedImage, setProcessedImage] = useState(null);
  const [detectedLabel, setDetectedLabel] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [holdProgress, setHoldProgress] = useState(0);
  const [bufferProgress, setBufferProgress] = useState(0);
  const [sentence, setSentence] = useState("");
  const [wordError, setWordError] = useState("");

  const socketRef = useRef(null);
  const isProcessingRef = useRef(false);
  const modeRef = useRef(mode);   // the frame loop reads this without re-subscribing

  useEffect(() => { modeRef.current = mode; }, [mode]);

  useEffect(() => {
    socketRef.current = io('http://localhost:5002');

    socketRef.current.on('connect', () => {
      console.log('Connected to ML Server');
    });

    socketRef.current.on('text_processed_frame', (data) => {
      if (data.image !== undefined) setProcessedImage(data.image);
      if (data.label !== undefined) setDetectedLabel(data.label);
      if (data.confidence !== undefined) setConfidence(data.confidence);
      if (data.holdProgress !== undefined) setHoldProgress(data.holdProgress);
      if (data.sentence !== undefined) setSentence(data.sentence);
      isProcessingRef.current = false;
    });

    socketRef.current.on('word_processed_frame', (data) => {
      if (data.error) setWordError(data.error);
      if (data.image !== undefined) setProcessedImage(data.image);
      if (data.label !== undefined) setDetectedLabel(data.label);
      if (data.confidence !== undefined) setConfidence(data.confidence);
      if (data.stableProgress !== undefined) setHoldProgress(data.stableProgress);
      if (data.bufferProgress !== undefined) setBufferProgress(data.bufferProgress);
      if (data.sentence !== undefined) setSentence(data.sentence);
      isProcessingRef.current = false;
    });

    return () => {
      socketRef.current.disconnect();
      stopCamera();
    };
  }, []);

  const switchMode = (next) => {
    if (next === mode) return;
    setMode(next);
    setDetectedLabel("");
    setConfidence(0);
    setHoldProgress(0);
    setBufferProgress(0);
    setWordError("");
    isProcessingRef.current = false;
  };

  // Space / Backspace drive word breaks and corrections directly,
  // instead of relying on the "space"/"del" gesture classes.
  useEffect(() => {
    const handleKeyDown = (e) => {
      const event = modeRef.current === 'words' ? 'word_key' : 'text_key';
      if (e.key === ' ' || e.code === 'Space') {
        // Word mode separates words automatically on commit, so Space is only
        // meaningful when fingerspelling.
        if (modeRef.current === 'words') return;
        e.preventDefault();
        socketRef.current?.emit('text_key', { action: 'space' });
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        socketRef.current?.emit(event, { action: 'backspace' });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const clearSentence = () => {
    socketRef.current?.emit(mode === 'words' ? 'word_key' : 'text_key', { action: 'clear' });
  };

  useEffect(() => {
    let animationId;
    let lastSendTime = 0;

    const sendFrame = (timestamp) => {
      if (isActive && videoRef.current && canvasRef.current) {
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
                    modeRef.current === 'words' ? 'word_frame' : 'text_frame',
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
    setConfidence(0);
    setHoldProgress(0);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">

        {/* Header */}
        <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-4">
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-blue-600">
                    Sign to Text
                </h2>

                {/* Which model reads the camera. Letters and words are separate
                    models, so this picks one rather than guessing. */}
                <div className="flex bg-gray-100 dark:bg-gray-800 rounded-xl p-1 gap-1">
                    {[['letters', 'Letters'], ['words', 'Words']].map(([key, text]) => (
                        <button
                            key={key}
                            onClick={() => switchMode(key)}
                            className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors ${
                                mode === key
                                    ? 'bg-white dark:bg-[#0f172a] text-indigo-500 shadow-sm'
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
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start h-[520px]">

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
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Start the camera and hold each letter steady to spell out words.</p>
                </div>
            ) : processedImage ? (
                <>
                    <img src={processedImage} alt="Live Stream" className="w-full h-full object-cover" />

                    <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
                        <div className="bg-black/70 backdrop-blur-md p-3 rounded-xl border border-white/10 text-white min-w-[200px]">
                            <p className="text-xs uppercase tracking-wider text-gray-300 mb-1">
                                {mode === 'words' ? 'Word sign' : 'Holding'}
                            </p>
                            <div className="flex items-end gap-3 mb-2">
                                <span className="text-3xl font-bold">{detectedLabel || "—"}</span>
                                <span className="text-indigo-300 font-mono mb-1">{Math.round(confidence * 100)}%</span>
                            </div>

                            {/* In word mode the model needs a full window of frames
                                before it can predict at all, so show that filling up
                                separately from the confirm-to-commit progress. */}
                            {mode === 'words' && bufferProgress < 1 && (
                                <div className="mb-2">
                                    <p className="text-[10px] text-gray-400 mb-1">reading movement...</p>
                                    <div className="w-full bg-white/10 h-1 rounded-full overflow-hidden">
                                        <div className="h-full bg-gray-400 transition-all duration-100"
                                             style={{ width: `${Math.round(bufferProgress * 100)}%` }}></div>
                                    </div>
                                </div>
                            )}

                            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-indigo-400 transition-all duration-100 ease-linear"
                                    style={{ width: `${Math.round(holdProgress * 100)}%` }}
                                ></div>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="flex flex-col items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mb-4"></div>
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

          {/* Side Panel: Translated text + controls */}
          <div className="md:col-span-4 flex flex-col h-full space-y-4">

            <div className="flex-1 rounded-2xl p-5 flex flex-col bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-800 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <h3 className="text-gray-500 dark:text-gray-400 font-medium tracking-wide text-sm uppercase">Translated Text</h3>
                    <button
                        onClick={clearSentence}
                        className="text-xs font-semibold text-gray-400 hover:text-red-500 transition-colors"
                    >
                        Clear
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto text-xl font-semibold text-gray-900 dark:text-white break-words">
                    {sentence || <span className="text-gray-400 dark:text-gray-600 font-normal text-base">Start signing to build a sentence...</span>}
                    {sentence && <span className="inline-block w-0.5 h-5 bg-indigo-400 ml-0.5 align-middle animate-pulse"></span>}
                </div>
                {wordError && (
                    <div className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded-lg p-2 mt-3">
                        {wordError}
                    </div>
                )}

                <p className="text-xs text-gray-400 mt-4 leading-relaxed">
                    {mode === 'words' ? (
                        <>Perform a whole word sign and hold it briefly. Words are added
                        automatically — press <span className="font-semibold text-gray-500 dark:text-gray-300">Backspace</span> to
                        remove the last one.</>
                    ) : (
                        <>Hold a letter steady to add it. Press <span className="font-semibold text-gray-500 dark:text-gray-300">Space</span> for
                        a word break and <span className="font-semibold text-gray-500 dark:text-gray-300">Backspace</span> to
                        remove the last letter.</>
                    )}
                </p>
            </div>

            <div className="pt-2">
                {!isActive ? (
                    <button
                        onClick={startCamera}
                        className="w-full bg-[var(--text-color)] text-[var(--bg-color)] py-3.5 px-4 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all shadow-md flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Start Camera
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

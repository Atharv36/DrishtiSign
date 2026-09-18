import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';

export default function TextToSign({ close }) {
  const [text, setText] = useState('');
  const [useLLM, setUseLLM] = useState(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [gloss, setGloss] = useState([]);
  const [simplified, setSimplified] = useState('');
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [queueLength, setQueueLength] = useState(0);
  const [currentKind, setCurrentKind] = useState('');
  const [error, setError] = useState('');
  const [isSimplifying, setIsSimplifying] = useState(false);

  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5002');

    socketRef.current.on('connect', () => {
      console.log('Connected to ML Server');
    });

    socketRef.current.on('sign_translate_started', (data) => {
      setIsSimplifying(false);
      setError('');
      setSimplified(data.simplified || '');
      setGloss(data.gloss || []);
      setQueueLength((data.queue || []).length);
      setCurrentIndex(-1);
      setIsPlaying(true);
    });

    socketRef.current.on('sign_processed_frame', (data) => {
      setProcessedImage(data.image);
      setCurrentIndex(data.index);
      setCurrentKind(data.kind);
    });

    socketRef.current.on('sign_sequence_done', () => {
      setIsPlaying(false);
    });

    socketRef.current.on('sign_translate_error', (data) => {
      setIsSimplifying(false);
      setIsPlaying(false);
      setError(data.message || 'Something went wrong.');
      if (data.simplified) setSimplified(data.simplified);
      if (data.gloss) setGloss(data.gloss);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  const handleTranslate = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setError('');
    setIsSimplifying(useLLM);
    socketRef.current.emit('translate_text', { text: trimmed, useLLM });
  };

  const handleStop = () => {
    socketRef.current.emit('stop_sign_sequence');
    setIsPlaying(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleTranslate();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">

        {/* Header */}
        <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-cyan-600">
                Text to Sign
            </h2>
            <button onClick={close} className="text-gray-500 hover:text-red-500 transition-colors p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start h-[560px]">

          {/* Avatar Feed */}
          <div className="md:col-span-7 bg-gray-100 dark:bg-black rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 shadow-inner relative h-full flex flex-col items-center justify-center">

            {processedImage ? (
                <>
                    <img src={processedImage} alt="Avatar performing sign" className="w-full h-full object-cover" />
                    {isPlaying && (
                        <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full flex items-center gap-2 border border-white/10">
                            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                            <span className="text-xs text-white font-medium tracking-wide">PLAYING</span>
                        </div>
                    )}
                    {currentIndex >= 0 && (
                        <div className="absolute bottom-4 left-4 right-4">
                            <div className="bg-black/70 backdrop-blur-md px-4 py-2 rounded-xl border border-white/10 text-white flex items-center justify-between">
                                <span className="text-sm">
                                    Sign {currentIndex + 1} / {queueLength}
                                    <span className="text-gray-400 ml-2">({currentKind === 'word' ? 'whole word' : 'fingerspelled'})</span>
                                </span>
                            </div>
                        </div>
                    )}
                </>
            ) : (
                <div className="text-center p-8 flex flex-col items-center">
                    <div className="w-20 h-20 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4 shadow-sm">
                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                    </div>
                    <p className="text-gray-500 dark:text-gray-400 font-medium text-lg">Avatar is idle</p>
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Type a sentence and translate it to watch the 3D avatar sign it.</p>
                </div>
            )}
          </div>

          {/* Input + Gloss Panel */}
          <div className="md:col-span-5 flex flex-col h-full space-y-4">

            <div className="flex flex-col">
                <label className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-1">Your sentence</label>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type something to sign..."
                    rows={4}
                    className="w-full bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-800 p-3 text-[var(--text-color)] placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-teal-400/50"
                />
                <label className="flex items-center gap-2 mt-2 text-xs text-gray-500 dark:text-gray-400 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={useLLM}
                        onChange={(e) => setUseLLM(e.target.checked)}
                        className="accent-teal-500"
                    />
                    Simplify with AI first (slower, ~15-20s, for long/complex sentences)
                </label>
            </div>

            <button
                onClick={handleTranslate}
                disabled={!text.trim() || isSimplifying}
                className="w-full bg-[var(--text-color)] text-[var(--bg-color)] py-3 px-4 rounded-xl font-bold tracking-wide hover:opacity-90 transition-all shadow-md flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
            >
                {isSimplifying ? (
                    <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                        Simplifying with AI...
                    </>
                ) : (
                    <>
                        Translate & Play
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </>
                )}
            </button>

            {isPlaying && (
                <button
                    onClick={handleStop}
                    className="w-full bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/20 hover:border-red-500 py-2.5 px-4 rounded-xl font-medium transition-all"
                >
                    Stop
                </button>
            )}

            {error && (
                <div className="text-sm text-red-500 bg-red-500/10 border border-red-500/20 rounded-xl p-3">
                    {error}
                </div>
            )}

            {gloss.length > 0 && (
                <div className="flex-1 rounded-2xl p-4 bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-800 overflow-y-auto">
                    {simplified && useLLM && (
                        <div className="mb-3">
                            <h4 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-1">Simplified</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-300">{simplified}</p>
                        </div>
                    )}
                    <h4 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-2">Signs to perform</h4>
                    <div className="flex flex-wrap gap-1.5">
                        {gloss.map((word, i) => (
                            <span
                                key={i}
                                className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-teal-500/10 text-teal-600 dark:text-teal-300 border border-teal-500/20"
                            >
                                {word}
                            </span>
                        ))}
                    </div>
                </div>
            )}

          </div>

        </div>

      </div>
    </div>
  );
}

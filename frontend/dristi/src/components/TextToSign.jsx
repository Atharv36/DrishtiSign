import React, { useRef, useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { SIGN_IMAGES } from '../constants/signs';

// How long each sign is shown. Kept here (not on the server) because playback
// is now entirely client-side - the server only does the language work.
const SIGN_HOLD_MS = 1300;

export default function TextToSign({ close }) {
  const [text, setText] = useState('');
  const [useLLM, setUseLLM] = useState(false);

  const [queue, setQueue] = useState([]);
  const [gloss, setGloss] = useState([]);
  const [simplified, setSimplified] = useState('');
  const [index, setIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSimplifying, setIsSimplifying] = useState(false);
  const [error, setError] = useState('');

  const socketRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5002');

    socketRef.current.on('sign_translate_started', (data) => {
      setIsSimplifying(false);
      setError('');
      setSimplified(data.simplified || '');
      setGloss(data.gloss || []);
      setQueue(data.queue || []);
      setIndex((data.queue || []).length ? 0 : -1);
      setIsPlaying(Boolean((data.queue || []).length));
    });

    socketRef.current.on('sign_translate_error', (data) => {
      setIsSimplifying(false);
      setIsPlaying(false);
      setError(data.message || 'Something went wrong.');
      if (data.gloss) setGloss(data.gloss);
    });

    return () => {
      socketRef.current.disconnect();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Client-side playback: step through the queue on a timer. Nothing is
  // streamed from the server, so this keeps working even if the ML server is
  // busy, and it needs no rendering backend at all.
  useEffect(() => {
    if (!isPlaying || queue.length === 0) return undefined;

    timerRef.current = setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= queue.length) {
          clearInterval(timerRef.current);
          setIsPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, SIGN_HOLD_MS);

    return () => clearInterval(timerRef.current);
  }, [isPlaying, queue]);

  const handleTranslate = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (timerRef.current) clearInterval(timerRef.current);
    setError('');
    setIndex(-1);
    setIsPlaying(false);
    setIsSimplifying(useLLM);
    socketRef.current.emit('translate_text', { text: trimmed, useLLM });
  };

  const handleReplay = () => {
    if (!queue.length) return;
    setIndex(0);
    setIsPlaying(true);
  };

  const handleStop = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setIsPlaying(false);
  };

  const current = index >= 0 && index < queue.length ? queue[index] : null;
  const currentImage = current ? SIGN_IMAGES[current.label] : null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 p-6 rounded-3xl w-full max-w-5xl shadow-2xl overflow-hidden relative flex flex-col transition-colors">

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

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start h-[560px]">

          {/* Sign display */}
          <div className="md:col-span-7 bg-gray-50 dark:bg-gray-900 rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 relative h-full flex flex-col items-center justify-center">
            {current ? (
                <>
                    {currentImage ? (
                        <img
                            src={currentImage}
                            alt={`Sign for ${current.word}`}
                            className="max-h-[75%] max-w-[80%] object-contain rounded-2xl shadow-lg"
                        />
                    ) : (
                        // No reference image for this sign yet - show the letter
                        // or word itself rather than nothing.
                        <div className="flex flex-col items-center">
                            <div className="text-8xl font-black text-gray-900 dark:text-white">
                                {current.label}
                            </div>
                            <p className="text-xs text-gray-400 mt-3">no reference image yet</p>
                        </div>
                    )}

                    <div className="absolute bottom-4 left-4 right-4">
                        <div className="bg-black/70 backdrop-blur-md px-4 py-2 rounded-xl border border-white/10 text-white flex items-center justify-between">
                            <span className="text-sm">
                                Sign {index + 1} / {queue.length}
                                <span className="text-gray-400 ml-2">
                                    ({current.kind === 'word' ? 'whole word' : `spelling "${current.word}"`})
                                </span>
                            </span>
                            {isPlaying && (
                                <span className="flex items-center gap-2 text-xs">
                                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                                    PLAYING
                                </span>
                            )}
                        </div>
                        <div className="w-full bg-white/10 h-1 rounded-full mt-2 overflow-hidden">
                            <div
                                className="h-full bg-teal-400 transition-all duration-300"
                                style={{ width: `${((index + 1) / queue.length) * 100}%` }}
                            ></div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="text-center p-8 flex flex-col items-center">
                    <div className="w-20 h-20 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center mb-4">
                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path></svg>
                    </div>
                    <p className="text-gray-500 dark:text-gray-400 font-medium text-lg">Nothing to sign yet</p>
                    <p className="text-gray-400 dark:text-gray-500 text-sm mt-2 max-w-xs">Type a sentence and translate it to step through the signs.</p>
                </div>
            )}
          </div>

          {/* Controls */}
          <div className="md:col-span-5 flex flex-col h-full space-y-4">
            <div className="flex flex-col">
                <label className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-1">Your sentence</label>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type something to sign..."
                    rows={4}
                    className="w-full bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-800 p-3 text-[var(--text-color)] placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-teal-400/50"
                />
                <label className="flex items-center gap-2 mt-2 text-xs text-gray-500 dark:text-gray-400 cursor-pointer">
                    <input type="checkbox" checked={useLLM} onChange={(e) => setUseLLM(e.target.checked)} className="accent-teal-500" />
                    Simplify with AI first (slower, ~15-20s, for long sentences)
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
                ) : 'Translate & Play'}
            </button>

            {queue.length > 0 && (
                <div className="flex gap-2">
                    <button onClick={handleReplay} className="flex-1 py-2.5 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-[var(--text-color)] transition-colors">
                        Replay
                    </button>
                    {isPlaying && (
                        <button onClick={handleStop} className="flex-1 py-2.5 rounded-xl bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/20 font-medium transition-all">
                            Stop
                        </button>
                    )}
                </div>
            )}

            {error && (
                <div className="text-sm text-red-500 bg-red-500/10 border border-red-500/20 rounded-xl p-3">{error}</div>
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
                            <span key={i} className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-teal-500/10 text-teal-600 dark:text-teal-300 border border-teal-500/20">
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

import signA from '../static/A.jpg';
import signB from '../static/B.jpg';
import signC from '../static/C.jpg';
import signD from '../static/D.jpg';

// Labels we never want to show as practice targets:
//  - del / nothing / space are meta classes, not real signs.
//  - J and Z are motion-based letters (they trace a shape in the air) that a
//    single-frame landmark classifier can't do reliably.
const EXCLUDED_LABELS = new Set(['del', 'nothing', 'space', 'J', 'Z']);

// Fallback list used before /labels loads (or if the ML server is offline).
// Matches the current alphabet model: A-Z minus the excluded letters.
export const SIGNS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
];

// Reference photos we currently have in src/static. Signs without a photo
// fall back to the big text glyph in Learning mode.
export const SIGN_IMAGES = { A: signA, B: signB, C: signC, D: signD };

// Pull the model's actual vocabulary from the ML server so the practice list
// always matches what the model was trained on. After retraining with the word
// dataset, the new words (Hello, Thankyou, ...) show up here automatically —
// no frontend change needed. Falls back to SIGNS if the server is unreachable.
// The word model's vocabulary (motion signs). Separate from the letter list
// because they're separate models - see /word-labels on the ML server.
export async function fetchWordSigns() {
  try {
    const res = await fetch('http://localhost:5002/word-labels');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const labels = await res.json();
    return Array.isArray(labels) ? labels : [];
  } catch (err) {
    console.warn('Could not load word labels from ML server:', err.message);
    return [];
  }
}

export async function fetchSigns() {
  try {
    const res = await fetch('http://localhost:5002/labels');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const labels = await res.json();
    const usable = labels.filter((l) => !EXCLUDED_LABELS.has(l));
    return usable.length ? usable : SIGNS;
  } catch (err) {
    console.warn('Could not load labels from ML server, using fallback list:', err.message);
    return SIGNS;
  }
}

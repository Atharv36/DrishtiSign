import { DEFAULT_LANGUAGE } from './language';

const ML_SERVER = 'http://localhost:5002';


const EXCLUDED_LABELS = new Set(['del', 'nothing', 'space', 'J', 'Z']);


export const SIGNS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
];


const aslLetterModules = import.meta.glob('../static/static_letters/*.png', { eager: true });
const ASL_LETTER_MEDIA = Object.fromEntries(
  Object.entries(aslLetterModules).map(([path, mod]) => [
    path.split('/').pop().replace('.png', ''),
    { src: mod.default, type: 'image' },
  ])
);

const aslWordModules = import.meta.glob('../static/WORDS/*.mp4', { eager: true });
const ASL_WORD_MEDIA = {};
for (const [path, mod] of Object.entries(aslWordModules)) {
  const stem = path.split('/').pop().replace(/0+\.mp4$/i, '').trim().replace(/\s+/g, '').toLowerCase();
  const media = { src: mod.default, type: 'video' };

  ASL_WORD_MEDIA[stem] = media;
  ASL_WORD_MEDIA[stem.charAt(0).toUpperCase() + stem.slice(1)] = media;
}

const ASL_MEDIA = { ...ASL_LETTER_MEDIA, ...ASL_WORD_MEDIA };

const islImageModules = import.meta.glob('../static/isl/*.jpg', { eager: true });
const ISL_MEDIA = Object.fromEntries(
  Object.entries(islImageModules).map(([path, mod]) => [
    path.split('/').pop().replace('.jpg', ''),
    { src: mod.default, type: 'image' },
  ])
);

export const SIGN_IMAGES = Object.fromEntries(
  Object.entries(ASL_LETTER_MEDIA).map(([label, media]) => [label, media.src])
);

export function signMedia(lang = DEFAULT_LANGUAGE) {
  return lang === 'isl' ? ISL_MEDIA : ASL_MEDIA;
}

export async function fetchWordSigns(lang = DEFAULT_LANGUAGE) {
  try {
    const res = await fetch(`${ML_SERVER}/word-labels?lang=${lang}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const labels = await res.json();
    return Array.isArray(labels) ? labels : [];
  } catch (err) {
    console.warn('Could not load word labels from ML server:', err.message);
    return [];
  }
}

export async function fetchSigns(lang = DEFAULT_LANGUAGE) {
  try {
    const res = await fetch(`${ML_SERVER}/labels?lang=${lang}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const labels = await res.json();
    const usable = labels.filter((l) => !EXCLUDED_LABELS.has(l));
    if (usable.length) return usable;
  } catch (err) {
    console.warn('Could not load labels from ML server:', err.message);
  }
  return lang === DEFAULT_LANGUAGE ? SIGNS : [];
}

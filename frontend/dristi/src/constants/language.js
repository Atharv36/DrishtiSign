// Which sign language the app is reading and showing.
//
// Stored in localStorage the same way ThemeToggle stores the theme - there's no
// Context provider in this app, and none is needed here: every consumer
// (Sign-to-Text, Text-to-Sign, Learning, Practice) is a full-screen modal that
// covers the navbar, so the dropdown can't be changed while one is open. Each
// modal reads the language once when it mounts, and a change applies to the
// next modal opened.

export const LANGUAGES = [
  { id: 'asl', label: 'ASL', name: 'American Sign Language' },
  { id: 'isl', label: 'ISL', name: 'Indian Sign Language' },
];

export const DEFAULT_LANGUAGE = 'asl';

const STORAGE_KEY = 'signLanguage';

export function getLanguage() {
  const stored = localStorage.getItem(STORAGE_KEY);
  return LANGUAGES.some((l) => l.id === stored) ? stored : DEFAULT_LANGUAGE;
}

export function setLanguage(id) {
  localStorage.setItem(STORAGE_KEY, id);
}

export function languageLabel(id) {
  return (LANGUAGES.find((l) => l.id === id) || LANGUAGES[0]).label;
}

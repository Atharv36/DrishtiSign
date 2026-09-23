import { useState } from 'react';
import { LANGUAGES, getLanguage, setLanguage } from '../constants/language';

// A native <select> rather than a custom menu: it's keyboard and screen-reader
// correct for free, and this app has no dropdown component to match.
const LanguageSelect = () => {
  const [lang, setLang] = useState(getLanguage);

  const handleChange = (e) => {
    setLanguage(e.target.value);
    setLang(e.target.value);
  };

  return (
    <select
      value={lang}
      onChange={handleChange}
      aria-label="Sign language"
      title="Which sign language the camera reads and the app teaches"
      className="px-2 py-2 rounded-md bg-transparent border border-black/10 dark:border-white/10 text-sm font-semibold text-accent hover:bg-black/5 dark:hover:bg-white/10 transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent/40"
    >
      {LANGUAGES.map((l) => (
        <option key={l.id} value={l.id} className="text-black">
          {l.label}
        </option>
      ))}
    </select>
  );
};

export default LanguageSelect;

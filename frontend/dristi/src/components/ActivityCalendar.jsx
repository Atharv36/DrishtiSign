import { useState, useEffect, useMemo } from 'react';
import { getActivity } from '../services/api';

const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

const toDateStr = (year, month, day) =>
  `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

export default function ActivityCalendar() {
  const [activeDates, setActiveDates] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [cursor, setCursor] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });

  useEffect(() => {
    getActivity()
      .then((dates) => setActiveDates(new Set(dates)))
      .catch((err) => console.error('Failed to load activity:', err))
      .finally(() => setLoading(false));
  }, []);

  const todayStr = useMemo(() => new Date().toLocaleDateString('en-CA'), []);

  const cells = useMemo(() => {
    const { year, month } = cursor;
    const firstWeekday = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    const list = Array.from({ length: firstWeekday }, () => null);
    for (let day = 1; day <= daysInMonth; day++) {
      list.push(day);
    }
    return list;
  }, [cursor]);

  const changeMonth = (delta) => {
    setCursor(({ year, month }) => {
      const next = new Date(year, month + delta, 1);
      return { year: next.getFullYear(), month: next.getMonth() };
    });
  };

  return (
    <div className="h-full bg-white dark:bg-[#0f172a] border border-gray-200 dark:border-gray-800 rounded-3xl p-6 shadow-sm flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg text-[var(--text-color)]">
          {MONTH_NAMES[cursor.month]} {cursor.year}
        </h3>
        <div className="flex gap-1">
          <button
            onClick={() => changeMonth(-1)}
            aria-label="Previous month"
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" /></svg>
          </button>
          <button
            onClick={() => changeMonth(1)}
            aria-label="Next month"
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" /></svg>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent-color)]"></div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-7 gap-1.5 mb-1.5">
            {WEEKDAYS.map((wd, i) => (
              <div key={i} className="text-center text-xs font-medium text-gray-400 uppercase">
                {wd}
              </div>
            ))}
          </div>

          <div className="grid grid-cols-7 gap-1.5">
            {cells.map((day, idx) => {
              if (day === null) return <div key={`pad-${idx}`} />;

              const dateStr = toDateStr(cursor.year, cursor.month, day);
              const isFuture = dateStr > todayStr;
              const isToday = dateStr === todayStr;
              const isActive = activeDates.has(dateStr);

              let content;
              if (isFuture) {
                content = <span className="text-gray-300 dark:text-gray-700">{day}</span>;
              } else if (isActive) {
                content = (
                  <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="3">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                );
              } else {
                content = (
                  <svg className="w-3.5 h-3.5 text-gray-300 dark:text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="3">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                );
              }

              return (
                <div
                  key={dateStr}
                  title={dateStr}
                  className={`aspect-square rounded-lg flex flex-col items-center justify-center gap-0.5 text-[10px] font-medium
                    ${isActive ? 'bg-emerald-50 dark:bg-emerald-900/20' : 'bg-gray-50 dark:bg-gray-800/40'}
                    ${isToday ? 'ring-2 ring-[var(--accent-color)]' : ''}
                  `}
                >
                  {!isFuture && <span className="text-gray-400 dark:text-gray-500">{day}</span>}
                  {content}
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-100 dark:border-gray-800 text-xs text-gray-500">
            <span className="flex items-center gap-1.5">
              <svg className="w-3 h-3 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
              Practiced
            </span>
            <span className="flex items-center gap-1.5">
              <svg className="w-3 h-3 text-gray-300 dark:text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="3"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              Missed
            </span>
          </div>
        </>
      )}
    </div>
  );
}

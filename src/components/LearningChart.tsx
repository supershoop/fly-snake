import { useMemo, useState } from 'react';

const WIDTH = 640, HEIGHT = 210, PAD = { left: 38, right: 118, top: 14, bottom: 28 };
const WINDOW = 20;  // games in the rolling average
/** Colour follows the wiring, never the order the runs happened in. Validated pair for the dark surface (dataviz validator: all checks pass). */
const WIRING = { real: { label: 'Real wiring', color: '#199e70' }, shuffled: { label: 'Scrambled wiring', color: '#d95926' } } as const;
export type LearningRun = { wiring: keyof typeof WIRING; scores: number[] };
const niceStep = (max: number) => [1, 2, 5, 10, 20, 50].find(step => max / step <= 5) ?? 100;
const rolling = (scores: number[]) => scores.map((_, i) => { const slice = scores.slice(Math.max(0, i + 1 - WINDOW), i + 1); return slice.reduce((a, b) => a + b, 0) / slice.length; });

/** Rolling average of food per finished game while the readout learns; one line per wiring trained in this session. */
export function LearningChart({ runs, current }: { runs: LearningRun[]; current: keyof typeof WIRING }) {
  const [hover, setHover] = useState<number | null>(null);
  const series = useMemo(() => runs.filter(run => run.scores.length >= 2).map(run => ({ ...run, ...WIRING[run.wiring], points: rolling(run.scores) })), [runs]);
  if (!series.length) return <p className="chart-empty">The learning curve appears after the first two finished games.</p>;
  const games = Math.max(...series.map(s => s.points.length)), yMax = Math.max(5, ...series.flatMap(s => s.points)) * 1.1, step = niceStep(yMax);
  const plotW = WIDTH - PAD.left - PAD.right, plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / Math.max(1, games - 1)) * plotW, y = (v: number) => PAD.top + plotH - (v / yMax) * plotH;
  const ticks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
  const labelY = series.map(s => y(s.points[s.points.length - 1])).map((value, i, all) => i && Math.abs(value - all[i - 1]) < 12 ? all[i - 1] + 12 : value);
  return <figure className="learning-chart">
    <figcaption>Food per game while learning <small>rolling average of the last {WINDOW} finished games, all flies · same learning rule for every line</small></figcaption>
    {series.length > 1 && <div className="chart-legend">{series.map(s => <span key={s.wiring}><i style={{ background: s.color }}/>{s.label}{s.wiring === current ? ' (training now)' : ''}</span>)}</div>}
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`Learning curves. ${series.map(s => `${s.label}: ${s.points[s.points.length - 1].toFixed(1)} after ${s.points.length} games`).join('. ')}`}
      onPointerMove={event => { const box = event.currentTarget.getBoundingClientRect(); const i = Math.round(((event.clientX - box.left) / box.width * WIDTH - PAD.left) / plotW * (games - 1)); setHover(Math.max(0, Math.min(games - 1, i))); }}
      onPointerLeave={() => setHover(null)}>
      {ticks.map(t => <g key={t}><line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className="chart-grid"/><text x={PAD.left - 6} y={y(t) + 3} textAnchor="end" className="chart-tick">{t}</text></g>)}
      <text x={PAD.left} y={HEIGHT - 6} className="chart-tick">game 1</text><text x={PAD.left + plotW} y={HEIGHT - 6} textAnchor="end" className="chart-tick">game {games}</text>
      {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} className="chart-crosshair"/>}
      {series.map((s, n) => { const last = s.points.length - 1, at = hover === null ? last : Math.min(hover, last); return <g key={s.wiring}>
        <path d={s.points.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"/>
        <circle cx={x(at)} cy={y(s.points[at])} r="4" fill={s.color} stroke="#101713" strokeWidth="2"/>
        <text x={PAD.left + plotW + 8} y={labelY[n] + 3} className="chart-value">{s.points[at].toFixed(1)} <tspan className="chart-tick">{s.label.split(' ')[0].toLowerCase()}{hover !== null ? ` · game ${at + 1}` : ''}</tspan></text>
      </g>; })}
    </svg>
    <details className="chart-table"><summary>Show as table</summary><table><thead><tr><th scope="col">After game</th>{series.map(s => <th key={s.wiring} scope="col">{s.label}</th>)}</tr></thead>
      <tbody>{Array.from({ length: games }, (_, i) => i).filter(i => i === games - 1 || (i + 1) % Math.max(10, Math.ceil(games / 12 / 10) * 10) === 0).map(i => <tr key={i}><td>{i + 1}</td>{series.map(s => <td key={s.wiring}>{s.points[i]?.toFixed(1) ?? '–'}</td>)}</tr>)}</tbody></table></details>
  </figure>;
}

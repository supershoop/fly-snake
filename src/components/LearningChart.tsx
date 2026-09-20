import { useMemo, useState } from 'react';

const WIDTH = 640, HEIGHT = 200, PAD = { left: 38, right: 96, top: 14, bottom: 28 };
const WINDOW = 20;          // games in the rolling average
const SERIES = '#199e70';   // single series: validated aqua for dark surfaces; text stays in text colours
const niceStep = (max: number) => [1, 2, 5, 10, 20, 50].find(step => max / step <= 5) ?? 100;

/** Rolling average of food per finished game while the readout learns. One series, so the title names it and there is no legend. */
export function LearningChart({ scores, reference }: { scores: number[]; reference?: { label: string; value: number } }) {
  const [hover, setHover] = useState<number | null>(null);
  const points = useMemo(() => scores.map((_, i) => {
    const slice = scores.slice(Math.max(0, i + 1 - WINDOW), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  }), [scores]);
  if (points.length < 2) return <p className="chart-empty">The learning curve appears after the first two finished games.</p>;
  const yMax = Math.max(5, reference?.value ?? 0, ...points) * 1.1, step = niceStep(yMax);
  const plotW = WIDTH - PAD.left - PAD.right, plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (points.length - 1)) * plotW, y = (v: number) => PAD.top + plotH - (v / yMax) * plotH;
  const path = points.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const last = points.length - 1, shown = hover ?? last;
  const ticks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
  return <figure className="learning-chart">
    <figcaption>Food per game while learning <small>rolling average of the last {WINDOW} finished games, all flies</small></figcaption>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`Learning curve: rolling average food per game is ${points[last].toFixed(1)} after ${points.length} games`}
      onPointerMove={event => { const box = event.currentTarget.getBoundingClientRect(); const i = Math.round(((event.clientX - box.left) / box.width * WIDTH - PAD.left) / plotW * (points.length - 1)); setHover(Math.max(0, Math.min(last, i))); }}
      onPointerLeave={() => setHover(null)}>
      {ticks.map(t => <g key={t}><line x1={PAD.left} x2={PAD.left + plotW} y1={y(t)} y2={y(t)} className="chart-grid"/><text x={PAD.left - 6} y={y(t) + 3} textAnchor="end" className="chart-tick">{t}</text></g>)}
      <text x={PAD.left} y={HEIGHT - 6} className="chart-tick">game 1</text><text x={PAD.left + plotW} y={HEIGHT - 6} textAnchor="end" className="chart-tick">game {points.length}</text>
      {reference && <g><line x1={PAD.left} x2={PAD.left + plotW} y1={y(reference.value)} y2={y(reference.value)} className="chart-reference"/>
        <text x={PAD.left + plotW + 6} y={y(reference.value) + 3} className="chart-tick">{reference.label} {reference.value}</text></g>}
      <path d={path} fill="none" stroke={SERIES} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"/>
      <line x1={x(shown)} x2={x(shown)} y1={PAD.top} y2={PAD.top + plotH} className="chart-crosshair" opacity={hover === null ? 0 : 1}/>
      <circle cx={x(shown)} cy={y(points[shown])} r="4" fill={SERIES} stroke="#101713" strokeWidth="2"/>
      <text x={Math.min(x(shown) + 8, PAD.left + plotW + 6)} y={Math.max(PAD.top + 10, y(points[shown]) - 8)} className="chart-value">{points[shown].toFixed(1)}{hover !== null ? ` · game ${shown + 1}` : ' now'}</text>
    </svg>
    <details className="chart-table"><summary>Show as table</summary><table><thead><tr><th scope="col">After game</th><th scope="col">Rolling average food</th></tr></thead>
      <tbody>{points.map((v, i) => ({ v, i })).filter(({ i }) => i === last || (i + 1) % Math.max(10, Math.ceil(points.length / 12 / 10) * 10) === 0).map(({ v, i }) => <tr key={i}><td>{i + 1}</td><td>{v.toFixed(1)}</td></tr>)}</tbody></table></details>
  </figure>;
}

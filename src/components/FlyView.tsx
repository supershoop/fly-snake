import { useMemo } from 'react';
import type { LiveFrame, VisionStatic } from '../lib/live';

const FOOD = '#dfb672', THREAT = '#e07a7a', DIM = '#2b3440';
const HEADINGS: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const STRIP_W = 720, STRIP_H = 150, HORIZON = 62;
const x = (azimuth: number) => STRIP_W / 2 + azimuth * (STRIP_W / 360);

type Seen = { azimuth: number; halfWidth: number; distance: number; food: boolean };

/** The same geometry flybrain/vision.py uses: where things are around the selected snake's head, as angles. */
function scene(frame: LiveFrame, range: number): Seen[] {
  const fly = frame.flies[frame.selected], arena = frame.arenas[fly.arena], snake = arena.snakes[fly.snake];
  const [headX, headY] = snake.body[0], [hx, hy] = HEADINGS[snake.heading];
  const look = (cx: number, cy: number, food: boolean): Seen => {
    const dx = cx - headX, dy = cy - headY, forward = dx * hx + dy * hy, rightward = dx * -hy + dy * hx, distance = Math.hypot(forward, rightward);
    return { azimuth: Math.atan2(rightward, forward) * 180 / Math.PI, halfWidth: Math.max(6, Math.atan2(.5, distance) * 180 / Math.PI), distance, food };
  };
  const blocked = new Set<string>();
  arena.snakes.forEach(other => { if (other.alive) (other === snake ? other.body.slice(1) : other.body).forEach(([cx, cy]) => blocked.add(`${cx},${cy}`)); });
  const seen: Seen[] = arena.foods.map(([fx, fy]) => look(fx, fy, true));
  const reach = Math.ceil(range);
  for (let cx = headX - reach; cx <= headX + reach; cx++) for (let cy = headY - reach; cy <= headY + reach; cy++) {
    const distance = Math.hypot(cx - headX, cy - headY), outside = cx < 0 || cy < 0 || cx >= arena.size || cy >= arena.size;
    if (distance > 0 && distance <= range && (outside || blocked.has(`${cx},${cy}`))) seen.push(look(cx, cy, false));
  }
  return seen.sort((a, b) => b.distance - a.distance);  // far first, so near things draw on top
}

/** Fly point of view (a flat world gives a 1-D horizon) and the two eyes with the detector cells that are lit. */
export function FlyView({ frame, vision, send }: { frame: LiveFrame | null; vision: VisionStatic | null; send: (message: object) => void }) {
  const eye = useMemo(() => {
    if (!vision) return null;
    const us = vision.eye.columns.map(c => c[0]), vs = vision.eye.columns.map(c => c[1]);
    return { uFront: Math.max(...us), uBack: Math.min(...us), vTop: Math.max(...vs), vBottom: Math.min(...vs) };
  }, [vision]);
  if (!frame || !vision || !eye) return null;
  const lit = new Map(frame.vision?.view ?? []), cells = vision.retina.cells, [, back] = vision.retina.fieldDeg;
  const retina = frame.encoder === 'retina';
  const EYE_W = 330, EYE_H = 190, GAP = 30, k = (EYE_W - 20) / (eye.uFront - eye.uBack), kv = (EYE_H - 20) / (eye.vTop - eye.vBottom);
  // front of each eye faces the middle of the picture, dorsal is up
  const ex = (u: number, side: string) => side === 'L' ? EYE_W - 10 - (eye.uFront - u) * k : EYE_W + GAP + 10 + (eye.uFront - u) * k;
  const ey = (v: number) => 10 + (eye.vTop - v) * kv;
  return <section className="model-status fly-view" aria-label="What the fly sees">
    <div className="fly-view-head"><strong>FLY POINT OF VIEW</strong>
      <div className="controls">
        <button aria-pressed={!retina} title="Five on/off channels: food left/right, threat left/right/ahead" onClick={() => send({ encoder: 'channels' })}>5 channels</button>
        <button aria-pressed={retina} title="Each detector cell is driven only by what lies in its own viewing direction, derived from the connectome" onClick={() => send({ encoder: 'retina' })}>Retinotopic eye</button>
      </div></div>
    <p>{retina ? 'The brain receives exactly this: each detector cell is driven only by what lies in the direction it looks.'
      : 'Shown for reference. In this mode the brain receives five on/off channels, not this picture. Switch to “Retinotopic eye” to feed it the picture.'}</p>
    <svg viewBox={`0 0 ${STRIP_W} ${STRIP_H}`} role="img" aria-label="Horizon around the snake's head, from behind-left to behind-right">
      <rect width={STRIP_W} height={STRIP_H} fill="#0c1117"/>
      <rect x={0} width={x(-back)} height={HORIZON * 2} fill="#080b0f"/><rect x={x(back)} width={STRIP_W - x(back)} height={HORIZON * 2} fill="#080b0f"/>
      <line x1={0} x2={STRIP_W} y1={HORIZON} y2={HORIZON} stroke="#28323e"/>
      {scene(frame, vision.retina.threatRange).map((o, i) => { const tall = Math.min(HORIZON - 4, 46 / o.distance);
        return <rect key={i} x={x(o.azimuth - o.halfWidth)} width={o.halfWidth * 2 * (STRIP_W / 360)} y={HORIZON - tall} height={tall * 2} rx={o.food ? tall : 1}
          fill={o.food ? FOOD : THREAT} opacity={o.food ? 1 : Math.min(1, .25 + .75 / o.distance)}/>; })}
      {[-180, -90, 0, 90, 180].map(a => <g key={a}><line x1={x(a)} x2={x(a)} y1={HORIZON * 2} y2={HORIZON * 2 + 5} stroke="#566170"/>
        <text x={Math.min(STRIP_W - 4, Math.max(4, x(a)))} y={STRIP_H - 2} fontSize="9" fill="#7d8c9e" textAnchor={a === -180 ? 'start' : a === 180 ? 'end' : 'middle'}>{a === 0 ? 'ahead' : a === -90 ? 'left' : a === 90 ? 'right' : 'behind'}</text></g>)}
      {cells.map(([azimuth, , , , food], i) => { const drive = lit.get(i) ?? 0;
        return <line key={i} x1={x(azimuth)} x2={x(azimuth)} y1={food ? HORIZON * 2 + 1 : HORIZON * 2 + 8} y2={food ? HORIZON * 2 + 7 : HORIZON * 2 + 14}
          stroke={drive ? (food ? FOOD : THREAT) : DIM} strokeWidth={drive ? 1.6 : .6} opacity={drive ? .4 + .6 * drive : 1}/>; })}
    </svg>
    <p className="fly-view-key"><span style={{ color: FOOD }}>■</span> food · object detectors LC10 (upper ticks) &nbsp; <span style={{ color: THREAT }}>■</span> walls and bodies within {vision.retina.threatRange} cells · looming detectors LC4, LPLC2 (lower ticks) &nbsp; dark = outside the eyes’ field</p>
    <svg viewBox={`0 0 ${EYE_W * 2 + GAP} ${EYE_H}`} role="img" aria-label="Left and right eye maps with the detector cells that are lit">
      {vision.eye.columns.map(([u, v, side], i) => <circle key={i} cx={ex(u, side)} cy={ey(v)} r="1.5" fill="#1b232d"/>)}
      {cells.map(([, u, v, side, food], i) => { const drive = lit.get(i); return drive ? <circle key={i} cx={ex(u, side)} cy={ey(v)} r={2 + 3 * drive} fill={food ? FOOD : THREAT} opacity=".85"/> : null; })}
      <text x={EYE_W - 10} y={EYE_H - 2} fontSize="9" fill="#7d8c9e" textAnchor="end">left eye · front →</text>
      <text x={EYE_W + GAP + 10} y={EYE_H - 2} fontSize="9" fill="#7d8c9e">← front · right eye</text>
    </svg>
    <p className="fly-view-key">Each grey dot is one of the eye’s ~880 columns (MaleCNS eye map). A lit dot is a detector cell, placed where it looks: the synapse-weighted position of the columns that feed it. Only cells near the horizon are used because the game is flat.</p>
  </section>;
}

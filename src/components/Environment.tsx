import type { LiveFrame } from '../lib/live';

const CELL = 32;
const SENSE_LABELS: Record<string, string> = { food_L: 'food left · LC10 L', food_R: 'food right · LC10 R', danger_L: 'threat left · LC4 L', danger_R: 'threat right · LC4 R', danger_ahead: 'threat ahead · LPLC2' };

/** Snake driven by the connectome simulation. The game only supplies sensory drive; every move comes from the brain's descending neurons. */
export function Environment({ frame }: { frame: LiveFrame | null }) {
  if (!frame) return <div className="environment"><p>Waiting for the brain server</p><span>Start it with: python -m uvicorn flybrain.server:app --port 8000</span></div>;
  const { snake, channels } = frame, extent = snake.size * CELL;
  return <div className="environment">
    <svg viewBox={`0 0 ${extent} ${extent}`} role="img" aria-label={`Snake game, score ${snake.score}`}>
      <defs><pattern id="grid" width={CELL} height={CELL} patternUnits="userSpaceOnUse"><path d={`M${CELL} 0H0V${CELL}`} fill="none" stroke="#242a30" strokeWidth=".5"/></pattern></defs>
      <rect width={extent} height={extent} fill="url(#grid)" stroke="#344150"/>
      <circle cx={(snake.food[0] + .5) * CELL} cy={(snake.food[1] + .5) * CELL} r={CELL * .32} fill="#dfb672"/>
      {snake.body.map(([x, y], index) => <rect key={index} x={x * CELL + 2} y={y * CELL + 2} width={CELL - 4} height={CELL - 4} rx="5"
        fill={index === 0 ? '#84d7ef' : '#527fa3'} opacity={snake.alive ? 1 - index * .02 : .35}/>)}
    </svg>
    <div className="senses" aria-label="Sensory neurons being stimulated">
      {Object.entries(channels).map(([name, level]) => <span key={name} className={level > 0 ? 'on' : ''}>{SENSE_LABELS[name] ?? name}</span>)}
    </div>
  </div>;
}

import type { ArenaState, LiveFrame } from '../lib/live';

const CELL = 32;
const SENSE_LABELS: Record<string, string> = { food_L: 'food left · LC10 L', food_R: 'food right · LC10 R', danger_L: 'threat left · LC4 L', danger_R: 'threat right · LC4 R', danger_ahead: 'threat ahead · LPLC2' };

function Board({ arena, selectedSnake, label, onSelect }: { arena: ArenaState; selectedSnake: number | null; label: string; onSelect?: () => void }) {
  const extent = arena.size * CELL;
  return <svg viewBox={`0 0 ${extent} ${extent}`} role="img" aria-label={label} className={selectedSnake !== null ? 'selected' : ''} onClick={onSelect}>
    <defs><pattern id="grid" width={CELL} height={CELL} patternUnits="userSpaceOnUse"><path d={`M${CELL} 0H0V${CELL}`} fill="none" stroke="#242a30" strokeWidth=".5"/></pattern></defs>
    <rect width={extent} height={extent} fill="url(#grid)" stroke="#344150"/>
    {arena.foods.map(([x, y], index) => <circle key={index} cx={(x + .5) * CELL} cy={(y + .5) * CELL} r={CELL * .32} fill="#dfb672"/>)}
    {arena.snakes.map((snake, s) => snake.body.map(([x, y], index) => <rect key={`${s}-${index}`} x={x * CELL + 2} y={y * CELL + 2} width={CELL - 4} height={CELL - 4} rx="5"
      fill={snake.kind === 'human' ? (index === 0 ? '#f0a3a3' : '#a35252') : s === selectedSnake ? (index === 0 ? '#84d7ef' : '#527fa3') : (index === 0 ? '#9aa9b8' : '#4a5663')}
      opacity={snake.alive ? 1 - Math.min(.5, index * .02) : .3}/>))}
  </svg>;
}

/** Snakes driven by the connectome simulation. The game only supplies sensory drive; every fly move comes from the brain's descending neurons. */
export function Environment({ frame, onSelect }: { frame: LiveFrame | null; onSelect: (fly: number) => void }) {
  if (!frame) return <div className="environment"><p>Waiting for the brain server</p><span>Start it with: python -m uvicorn flybrain.server:app --port 8000</span></div>;
  const chosen = frame.flies[frame.selected];
  return <div className="environment">
    <div className={`boards boards-${frame.arenas.length > 1 ? 'many' : 'one'}`}>
      {frame.arenas.map((arena, a) => <Board key={a} arena={arena} selectedSnake={chosen?.arena === a ? chosen.snake : null}
        label={`Board ${a + 1}, scores ${arena.snakes.map(s => s.score).join(', ')}`}
        onSelect={() => { const fly = frame.flies.findIndex(f => f.arena === a); if (fly >= 0) onSelect(fly); }}/>)}
    </div>
    {chosen && <div className="senses" aria-label="Sensory neurons being stimulated in the selected fly">
      {Object.entries(chosen.channels).map(([name, level]) => <span key={name} className={level > 0 ? 'on' : ''}>{SENSE_LABELS[name] ?? name}</span>)}
    </div>}
  </div>;
}

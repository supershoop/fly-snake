import { useId } from 'react';
import type { ArenaState, LiveFrame, LiveStatus } from '../lib/live';
import { Icon } from './Icon';

const CELL = 32;
const SENSE_LABELS: Record<string, string> = { food_L: 'Food L', food_R: 'Food R', danger_L: 'Threat L', danger_R: 'Threat R', danger_ahead: 'Threat ahead' };
const COLORS = ['#8bbab1', '#cfa1b6', '#a7b4db', '#d9bb85', '#bba8d7', '#87bcc7', '#c4c58d', '#d89f82'];

function Board({ arena, selectedSnake, label }: { arena: ArenaState; selectedSnake: number | null; label: string }) {
  const id = useId();
  const extent = arena.size * CELL;
  return <svg viewBox={`0 0 ${extent} ${extent}`} role="img" aria-label={label}>
    <defs><pattern id={id} width={CELL} height={CELL} patternUnits="userSpaceOnUse"><path d={`M${CELL} 0H0V${CELL}`} fill="none" stroke="#28332e" strokeWidth=".65"/></pattern></defs>
    <rect width={extent} height={extent} fill="#101713"/>
    <rect width={extent} height={extent} fill={`url(#${id})`}/>
    {arena.foods.map(([x, y], index) => <g key={index}><circle cx={(x + .5) * CELL} cy={(y + .5) * CELL} r={CELL * .35} fill="#e4bc77" opacity=".1"/><circle cx={(x + .5) * CELL} cy={(y + .5) * CELL} r={CELL * .2} fill="#e4bc77"/></g>)}
    {arena.snakes.map((snake, s) => snake.body.map(([x, y], index) => <rect key={`${s}-${index}`} x={x * CELL + 2} y={y * CELL + 2} width={CELL - 4} height={CELL - 4} rx="5"
      fill={snake.kind === 'human' ? (index === 0 ? '#f0a59b' : '#b7716b') : s === selectedSnake ? (index === 0 ? '#d6f4ad' : '#97bf88') : COLORS[s % COLORS.length]}
      stroke={index === 0 ? '#f2f5e8' : 'none'} strokeWidth="1.4" opacity={snake.alive ? 1 - Math.min(.45, index * .02) : .3}/>))}
  </svg>;
}

/** Every fly move comes from the simulated brain's descending neurons. */
export function Environment({ frame, status, paused, onSelect, onHuman, onResume }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; onSelect: (fly: number) => void; onHuman: (direction: string) => void; onResume: () => void;
}) {
  if (!frame) return <div className="environment-empty">
    <div className="empty-symbol"><Icon name="snake" size={36}/></div>
    <span className="eyebrow">{status === 'live' ? 'Waiting for a frame' : 'Anatomy is ready to explore'}</span>
    <h3>{status === 'live' ? 'The brain is getting ready.' : 'Connect a brain. Start a game.'}</h3>
    <p>{status === 'live' ? 'The simulation may be initializing or paused. Resume the session if it was left paused.' : 'The game will appear here when the simulation connects. The anatomy views remain available.'}</p>
    {status === 'live' ? <button className="primary-button" onClick={onResume}><Icon name="play"/>Resume session</button> : <details className="connection-help"><summary>Start the brain server</summary><code>python -m uvicorn flybrain.server:app --port 8000</code><span>Reconnecting automatically. See the README for setup.</span></details>}
  </div>;
  const chosen = frame.flies[frame.selected];
  const snake = chosen && frame.arenas[chosen.arena]?.snakes[chosen.snake];
  const human = frame.layout === 'versus' ? frame.arenas[0]?.snakes.find(item => item.kind === 'human') : undefined;
  const many = frame.arenas.length > 1;
  return <div className="environment">
    <div className="game-scorebar">
      <div className="selected-fly"><span className="eyebrow">Inspecting</span>{frame.flies.length > 1 ? <select aria-label="Selected fly" value={frame.selected} onChange={event => onSelect(Number(event.target.value))}>{frame.flies.map((_, index) => <option key={index} value={index}>Fly {String(index + 1).padStart(2, '0')}</option>)}</select> : <strong>Fly 01</strong>}</div>
      <div><span className="eyebrow">Food eaten</span><strong className="score-value">{snake?.score ?? '—'}</strong></div>
      <div><span className="eyebrow">{human ? 'Your score' : 'Last game'}</span><strong className={human ? 'human-score' : ''}>{human ? human.score : snake?.games ? snake.lastScore : '—'}</strong></div>
      <div><span className="eyebrow">Games</span><strong>{snake?.games ?? '—'}</strong></div>
    </div>
    <div className={`board-stage ${many ? 'swarm-stage' : ''}`}>
      <div className={`boards boards-${many ? 'many' : 'one'}`}>
        {frame.arenas.map((arena, a) => {
          const selected = chosen?.arena === a;
          const label = `Board ${a + 1}, scores ${arena.snakes.map(s => s.score).join(', ')}`;
          const board = <Board arena={arena} selectedSnake={selected ? chosen.snake : null} label={label}/>;
          return many ? <button key={a} className="board-select" aria-label={`Inspect fly ${a + 1}, score ${arena.snakes[0]?.score ?? 0}`} aria-pressed={selected} onClick={() => { const fly = frame.flies.findIndex(f => f.arena === a); if (fly >= 0) onSelect(fly); }}>
            {board}<span><span>Fly {String(a + 1).padStart(2, '0')}</span><strong>{arena.snakes[0]?.score ?? 0}</strong></span>
          </button> : <div key={a} className="single-board">{board}</div>;
        })}
      </div>
      {(paused || frame.manual) && <div className="game-overlay"><span>{paused ? 'Simulation paused' : 'Sensory override'}<small>{paused ? 'Resume to continue the game' : 'Release the input to continue'}</small></span></div>}
    </div>
    <div className="board-legend"><span><i className="legend-fly"/>Selected fly</span>{human && <span><i className="legend-human"/>You</span>}<span><i className="legend-food"/>Food</span><span className="board-size">{frame.arenas[0]?.size} × {frame.arenas[0]?.size}</span></div>
    {frame.layout === 'versus' && <div className="human-controls" role="group" aria-label="Steer your snake"><span>Arrow keys / WASD</span>{[['left', '←'], ['up', '↑'], ['down', '↓'], ['right', '→']].map(([direction, symbol]) => <button key={direction} disabled={paused || frame.manual} aria-label={`Move ${direction}`} onClick={() => onHuman(direction)}>{symbol}</button>)}</div>}
    {chosen && <div className="senses" aria-label="Sensory inputs for the selected fly"><span className="senses-label">Senses</span>{Object.entries(chosen.channels).map(([name, level]) => <span key={name} className={level > 0 ? 'on' : ''} title={`${name}: ${Math.round(level * 100)}%`}><i aria-hidden="true"/>{SENSE_LABELS[name] ?? name}<span className="sr-only"> {Math.round(level * 100)}%</span></span>)}</div>}
  </div>;
}

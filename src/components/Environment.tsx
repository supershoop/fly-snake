import { useId } from 'react';
import type { ArenaState, LiveFrame, LiveStatus, PendingCommand } from '../lib/live';
import { Icon } from './Icon';

const CELL = 32;
const SENSE_LABELS: Record<string, string> = { food_L: 'Food L', food_R: 'Food R', danger_L: 'Threat L', danger_R: 'Threat R', danger_ahead: 'Threat ahead' };
function Board({ arena, label, onSnake }: { arena: ArenaState; label: string; onSnake?: (snake: number) => void }) {
  const id = useId();
  const extent = arena.size * CELL;
  return <svg viewBox={`0 0 ${extent} ${extent}`} role="img" aria-label={label}>
    <defs><pattern id={id} width={CELL * 2} height={CELL * 2} patternUnits="userSpaceOnUse">
      <rect width={CELL * 2} height={CELL * 2} fill="var(--ink)"/>
      <rect width={CELL} height={CELL} fill="var(--blue)" opacity=".22"/>
      <rect x={CELL} y={CELL} width={CELL} height={CELL} fill="var(--blue)" opacity=".22"/>
      <path d={`M${CELL} 0V${CELL * 2}M0 ${CELL}H${CELL * 2}M0 0H${CELL * 2}V${CELL * 2}H0Z`} fill="none" stroke="var(--cream)" strokeOpacity=".15" strokeWidth=".65"/>
    </pattern></defs>
    <rect width={extent} height={extent} fill={`url(#${id})`}/>
    {arena.foods.map(([x, y], index) => <g key={index}><circle cx={(x + .5) * CELL} cy={(y + .5) * CELL} r={CELL * .35} fill="#fff" opacity=".22"/><circle cx={(x + .5) * CELL} cy={(y + .5) * CELL} r={CELL * .2} fill="#fff"/></g>)}
    {arena.snakes.map((snake, s) => {
      const [headX, headY] = snake.body[0] ?? [0, 0];
      const bodyColor = snake.kind === 'human' ? 'var(--player-snake)' : 'var(--mauve)';
      const headAsset = snake.kind === 'human' ? '/player-head.svg' : '/fly-head.svg';
      const points = snake.body.map(([x, y]) => `${(x + .5) * CELL},${(y + .5) * CELL}`).join(' ');
      const interactive = onSnake && snake.kind === 'fly';
      const opacity = snake.alive ? 1 : .3;
      const headCenterX = (headX + .5) * CELL, headCenterY = (headY + .5) * CELL;
      return <g key={s} onClick={interactive ? () => onSnake(s) : undefined} style={interactive ? { cursor: 'pointer' } : undefined} opacity={opacity}>
        {/* One rounded stroke makes adjacent grid cells read as a single moving body. */}
        {snake.body.length > 1 && <polyline points={points} fill="none" stroke={bodyColor} strokeWidth={CELL - 4} strokeLinecap="round" strokeLinejoin="round"/>}
        <image href={headAsset} x={headX * CELL} y={headY * CELL} width={CELL} height={CELL} transform={`rotate(${snake.heading * 90} ${headCenterX} ${headCenterY})`} preserveAspectRatio="xMidYMid meet" pointerEvents="none" aria-hidden="true"/>
        {snake.body.length > 1 && <polyline points={points} fill="none" stroke="var(--cream)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" opacity=".2"/>}
      </g>;
    })}
  </svg>;
}

/** Every fly move comes from the simulated brain's descending neurons. */
export function Environment({ frame, status, paused, pending, picked, onPick, onSelect, onHuman, onResume }: {
  frame: LiveFrame | null; status: LiveStatus; paused: boolean; pending: PendingCommand | null; picked: number[]; onPick: (fly: number) => void; onSelect: (fly: number) => void; onHuman: (direction: string) => void; onResume: () => void;
}) {
  if (!frame) return <div className="environment-empty">
    <div className="empty-symbol">{status === 'live' ? <i className="spinner"/> : <Icon name="snake" size={36}/>}</div>
    <span className="eyebrow">{status === 'live' ? 'Waiting for a frame' : 'Anatomy is ready to explore'}</span>
    <h3>{status === 'live' ? 'The brain is getting ready.' : 'Connect a brain. Start a game.'}</h3>
    <p>{status === 'live' ? 'The simulation may be initializing or paused. Resume the session if it was left paused.' : 'The game will appear here when the simulation connects. The anatomy views remain available.'}</p>
    {status === 'live' ? <button className="primary-button" onClick={onResume}><Icon name="play"/>Resume session</button> : <details className="connection-help"><summary>Start the brain server</summary><code>python -m uvicorn flybrain.server:app --port 8000</code><span>Reconnecting automatically. See the README for setup.</span></details>}
  </div>;
  const selectedFly = pending?.selected ?? frame.selected;
  const chosen = frame.flies[selectedFly];
  const snake = chosen && frame.arenas[chosen.arena]?.snakes[chosen.snake];
  const human = frame.layout === 'versus' ? frame.arenas[0]?.snakes.find(item => item.kind === 'human') : undefined;
  const many = frame.arenas.length > 1;
  return <div className="environment">
    <div className="game-scorebar">
      <div className="selected-fly"><span className="eyebrow">Inspecting</span>{frame.flies.length > 1 ? <select aria-label="Selected fly" value={selectedFly} onChange={event => onSelect(Number(event.target.value))}>{frame.flies.map((_, index) => <option key={index} value={index}>Fly {String(index + 1).padStart(2, '0')}</option>)}</select> : <strong>Fly 01</strong>}</div>
      <div><span className="eyebrow">Food eaten</span><strong className="score-value">{snake?.score ?? '—'}</strong></div>
      <div><span className="eyebrow">{human ? 'Your score' : 'Last game'}</span><strong className={human ? 'human-score' : ''}>{human ? human.score : snake?.games ? snake.lastScore : '—'}</strong></div>
      <div><span className="eyebrow">Games</span><strong>{snake?.games ?? '—'}</strong></div>
      <div><span className="eyebrow">High score</span><strong>{snake?.highScore ?? snake?.score ?? '—'}</strong></div>
    </div>
    <div className={`board-stage ${many ? 'swarm-stage' : ''}`}>
      <div className="game-window">
        <div className="game-window-bar"><span>snake flies.exe</span><span className="window-controls" aria-hidden="true"><i>−</i><i>□</i><i>×</i></span></div>
        <div className={`boards boards-${many ? 'many' : 'one'}`}>
        {frame.arenas.map((arena, a) => {
          const label = `Board ${a + 1}, scores ${arena.snakes.map(s => s.score).join(', ')}`;
          const flyOf = (snake: number) => frame.flies.findIndex(f => f.arena === a && f.snake === snake);
          const boardFlies = arena.snakes.map((_, s) => flyOf(s)).filter(f => f >= 0);
          const isPicked = boardFlies.some(f => picked.includes(f)), isLesioned = boardFlies.some(f => frame.flies[f].lesion.length > 0);
          const board = <Board arena={arena} label={label} onSnake={many ? undefined : snake => { const f = flyOf(snake); if (f >= 0) onPick(f); }}/>;
          return many ? <button key={a} className={`board-select ${isLesioned ? 'is-lesioned' : ''}`} aria-label={`${isPicked ? 'Unpick' : 'Pick'} fly ${a + 1}, score ${arena.snakes[0]?.score ?? 0}${isLesioned ? ', lesioned' : ''}`} aria-pressed={isPicked} onClick={() => { if (boardFlies.length) onPick(boardFlies[0]); }}>
            {board}<span><span>Fly {String(a + 1).padStart(2, '0')}{isLesioned ? ' · lesioned' : ''}</span><strong>{arena.snakes[0]?.score ?? 0}</strong></span>
          </button> : <div key={a} className={`single-board ${isPicked ? 'is-picked' : ''} ${isLesioned ? 'is-lesioned' : ''}`}>{board}</div>;
        })}
        </div>
      </div>
      {(paused || frame.manual || (pending && (pending.layout !== undefined || pending.policy !== undefined || pending.wiring !== undefined))) && <div className={`game-overlay ${pending ? 'is-pending' : ''}`}><span>{pending ? <><i className="spinner"/>{pending.message}<small>Updating the simulation…</small></> : paused ? <>Simulation paused<small>Resume to continue the game</small></> : <>Sensory override<small>Release the input to continue</small></>}</span></div>}
    </div>
    <div className="board-legend"><span><i className="legend-fly"/>Selected fly</span><span className="pick-hint">{many ? 'Click boards to pick flies for the lesion lab' : 'Click a fly to open the lesion lab'}</span>{human && <span><i className="legend-human"/>You</span>}<span><i className="legend-food"/>Food</span><span className="board-size">{frame.arenas[0]?.size} × {frame.arenas[0]?.size}</span></div>
    {frame.layout === 'versus' && <div className="human-controls" role="group" aria-label="Steer your snake"><span>Arrow keys / WASD</span>{[['left', '←'], ['up', '↑'], ['down', '↓'], ['right', '→']].map(([direction, symbol]) => <button key={direction} disabled={paused || frame.manual} aria-label={`Move ${direction}`} onClick={() => onHuman(direction)}>{symbol}</button>)}</div>}
    {chosen && <div className="senses" aria-label="Sensory inputs for the selected fly"><span className="senses-label">Senses</span>{Object.entries(chosen.channels).map(([name, level]) => <span key={name} className={level > 0 ? 'on' : ''} title={`${name}: ${Math.round(level * 100)}%`}><i aria-hidden="true"/>{SENSE_LABELS[name] ?? name}<span className="sr-only"> {Math.round(level * 100)}%</span></span>)}{chosen.event && <span className={`on felt felt-${chosen.event}`} title={chosen.event === 'taste' ? 'Positive event stimulus: sugar taste neurons are stimulated, driving the feeding pathway' : 'Negative event stimulus: heat sensors are stimulated, driving the punishment pathway'}><i aria-hidden="true"/>{chosen.event === 'taste' ? 'Tasting sugar' : 'Pain'}</span>}</div>}
  </div>;
}

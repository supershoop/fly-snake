import { useEffect, useState } from 'react';
import type { LiveFrame } from '../lib/live';

/** Human leaderboard for the versus layout. One round = one human life; the fly's score in that round sits beside it. */
export function Leaderboard({ frame, send }: { frame: LiveFrame; send: (message: object) => void }) {
  const board = frame.leaderboard;
  const [name, setName] = useState(() => { try { return localStorage.getItem('fly-snake-player') ?? ''; } catch { return ''; } });
  useEffect(() => {
    const id = window.setTimeout(() => {
      send({ player: name });
      try { localStorage.setItem('fly-snake-player', name); } catch { /* private window: the name just is not remembered */ }
    }, 300);
    return () => window.clearTimeout(id);
  }, [name, send]);
  if (!board) return null;
  const human = frame.arenas[0]?.snakes.find(snake => snake.kind === 'human'), fly = frame.arenas[0]?.snakes.find(snake => snake.kind === 'fly');
  return <section className="model-status leaderboard" aria-label="Human leaderboard">
    <div><strong>YOU VS THE FLY</strong>
      <p><label>Your name <input value={name} maxLength={16} onChange={event => setName(event.target.value)} placeholder="anonymous" spellCheck={false}/></label></p>
      <p className="round-score"><span className="human-score">{human?.score ?? 0}</span> you · fly <span>{fly?.score ?? 0}</span></p>
      <p>A round lasts until you crash. Then your score goes on the board and both snakes start again.</p>
      <p>{board.rounds ? `${board.rounds} rounds played · humans won ${board.humanWins} · the fly won ${board.flyWins} · ${board.rounds - board.humanWins - board.flyWins} draws` : 'No rounds played yet.'}</p>
    </div>
    <div><strong>TOP HUMANS</strong>
      {board.top.length ? <table><thead><tr><th scope="col">#</th><th scope="col">Name</th><th scope="col">You</th><th scope="col">Fly</th><th scope="col">Against</th></tr></thead>
        <tbody>{board.top.map((round, index) => <tr key={`${round.when}-${index}`} className={round.name === board.player && round.name !== 'anonymous' ? 'mine' : ''}>
          <td>{index + 1}</td><td>{round.name}</td><td>{round.human}</td><td>{round.fly}</td><td>{round.brain}</td></tr>)}</tbody></table>
        : <p>Be the first on the board.</p>}
    </div>
  </section>;
}

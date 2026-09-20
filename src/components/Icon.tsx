type Name = 'snake' | 'solo' | 'swarm' | 'versus' | 'arena' | 'pause' | 'play' | 'arrow' | 'brain' | 'sliders';

const paths: Record<Name, string> = {
  snake: 'M4 7h10a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h10M4 4v6m15 6v6',
  solo: 'M7 7h10v10H7z',
  swarm: 'M3 3h6v6H3zM15 3h6v6h-6zM3 15h6v6H3zM15 15h6v6h-6z',
  versus: 'M8 4 3 9l5 5M3 9h12M16 10l5 5-5 5M21 15H9',
  arena: 'M3 9V3h6M15 3h6v6M21 15v6h-6M9 21H3v-6M9 9h6v6H9z',
  pause: 'M8 5v14M16 5v14',
  play: 'm8 5 11 7-11 7Z',
  arrow: 'M4 12h16m-6-6 6 6-6 6',
  brain: 'M12 5c-4-5-9 0-6 4-5 2-3 9 1 8 0 4 5 4 5 0V5Zm0 0c4-5 9 0 6 4 5 2 3 9-1 8 0 4-5 4-5 0M8 10l4 2 4-2',
  sliders: 'M4 7h6m4 0h6M4 17h10m4 0h2M10 4v6M14 14v6',
};

export function Icon({ name, size = 18 }: { name: Name; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/></svg>;
}

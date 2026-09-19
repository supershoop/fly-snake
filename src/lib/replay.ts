export type ActivityFrame = { time: number; values: [number, number][] };
export type ModelReplay = {
  version: 1;
  dataset: 'male-cns:v1.0';
  source: { kind: 'synthetic' | 'predicted' | 'measured'; name: string; normalization: string };
  frames: ActivityFrame[];
};
const record = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

/** Validate uploaded output against visible body IDs, not array indices or coordinates. */
export function parseReplay(input: unknown, visibleIds: ReadonlySet<number>): ModelReplay {
  if (!record(input) || input.version !== 1 || input.dataset !== 'male-cns:v1.0') throw Error('Expected replay version 1 for male-cns:v1.0.');
  const source = input.source;
  if (!record(source) || !['synthetic', 'predicted', 'measured'].includes(String(source.kind)) || typeof source.name !== 'string' || !source.name.trim() || typeof source.normalization !== 'string' || !source.normalization.trim()) throw Error('Declare source kind, name and normalization.');
  if (!Array.isArray(input.frames) || input.frames.length < 2 || input.frames.length > 10000) throw Error('Supply 2 to 10,000 frames.');
  let previous = -1;
  for (const frame of input.frames) {
    if (!record(frame) || typeof frame.time !== 'number' || !Number.isFinite(frame.time) || frame.time < 0 || frame.time <= previous) throw Error('Frame times must be finite, nonnegative and strictly increasing.');
    previous = frame.time;
    if (!Array.isArray(frame.values) || frame.values.length > visibleIds.size) throw Error('Invalid frame values.');
    const seen = new Set<number>();
    for (const pair of frame.values) {
      if (!Array.isArray(pair) || pair.length !== 2 || !Number.isSafeInteger(pair[0]) || !visibleIds.has(pair[0])) throw Error('Every value must use a visible MaleCNS brain body ID.');
      if (seen.has(pair[0])) throw Error('Duplicate body ID in a frame.');
      if (typeof pair[1] !== 'number' || !Number.isFinite(pair[1]) || pair[1] < 0 || pair[1] > 1) throw Error('Activity values must be finite and normalized to [0, 1].');
      seen.add(pair[0]);
    }
  }
  if (input.frames[0].time !== 0) throw Error('The first frame must start at time 0.');
  return input as ModelReplay;
}

/** Hold the last sample; no fabricated interpolation, noise or spikes. */
export function frameAt(replay: ModelReplay, time: number): ActivityFrame {
  let low = 0, high = replay.frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (replay.frames[middle].time <= time) low = middle; else high = middle - 1;
  }
  return replay.frames[low];
}

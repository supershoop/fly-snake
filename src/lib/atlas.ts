export type Atlas = { positions: Float32Array; ids: Uint32Array; groups: Uint8Array; visibleIds: Set<number> };
export const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;
export async function loadAtlas(signal: AbortSignal): Promise<Atlas> {
  const read = async (path: string) => {
    const response = await fetch(asset(`data/brain-atlas/${path}`), { signal });
    if (!response.ok) throw Error(`Atlas could not load: ${path}`);
    return response;
  };
  const manifest = await (await read('manifest.json')).json();
  const buffers = await Promise.all(['positions.bin', 'ids.bin', 'groups.bin'].map(async name => (await read(name)).arrayBuffer()));
  const positions = new Float32Array(buffers[0]), ids = new Uint32Array(buffers[1]), groups = new Uint8Array(buffers[2]);
  if (positions.length !== manifest.count * 3 || ids.length !== manifest.count || groups.length !== manifest.count) throw Error('Atlas file lengths do not match the manifest.');
  const visibleIds = new Set(Array.from(ids).filter((_, index) => groups[index] < 3));
  if (visibleIds.size !== manifest.brainCount) throw Error('Brain selection does not match the manifest.');
  return { positions, ids, groups, visibleIds };
}

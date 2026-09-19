import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root = new URL('../public/', import.meta.url);
for (const [directory,manifestFile,field] of [['brain-atlas','manifest.json','exportSha256'],['flybody','checksums.json','sha256']]) {
  const manifest=JSON.parse(await readFile(new URL(`data/${directory}/${manifestFile}`,root),'utf8'));
  for(const [name,digest] of Object.entries(manifest[field])) {
    const bytes=await readFile(new URL(`data/${directory}/${name}`,root));
    if(createHash('sha256').update(bytes).digest('hex')!==digest) throw Error(`Asset checksum mismatch: ${directory}/${name}`);
  }
}
console.log('Anatomical asset hashes verified.');

import { asset } from '../lib/atlas';

/** Required by LICENSE. You may restyle or relocate this credit, but keep it readable and linked. */
export function Attribution() {
  return <footer><span>Built with <a href="https://github.com/cobanov/fly-connectome-template">fly-connectome-template</a> by <a href="https://github.com/cobanov">Mert Cobanov</a>. <a href={asset("TEMPLATE-LICENSE.txt")}>License</a></span><span>Data: <a href="https://male-cns.janelia.org/">MaleCNS · CC BY 4.0</a> · Body: <a href="https://github.com/TuragaLab/flybody">Flybody · Apache 2.0</a></span></footer>;
}

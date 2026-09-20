import { asset } from '../lib/atlas';

/** Required by LICENSE. You may restyle or relocate this credit, but keep it readable and linked. */
export function Attribution() {
  return <footer>
    <div className="footer-team">
      <span>Built by Wenya, Hang, Owen, and Kyle for Hack the North 2026.</span>
      <span>We’re sincerely sorry to this simulated fly for the horrifying little existence we made it live.</span>
    </div>
    <div className="footer-credits">
      <span>Built with <a href="https://github.com/cobanov/fly-connectome-template">fly-connectome-template</a> by <a href="https://github.com/cobanov">Mert Cobanov</a>. <a href={asset("TEMPLATE-LICENSE.txt")}>License</a></span>
      <span>Fly brain data: <a href="https://male-cns.janelia.org/">MaleCNS · CC BY 4.0</a>, from FlyEM / HHMI Janelia, the University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Body: <a href="https://github.com/TuragaLab/flybody">Flybody · Apache 2.0</a>.</span>
    </div>
  </footer>;
}

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ActivityFrame } from "../lib/replay";
import type { Atlas } from "../lib/atlas";

/** Real anatomy; model values are looked up by body ID, never by spatial proximity. */
const INITIAL_ZOOM = 2.1;

export function BrainScene({ atlas, frame, silenced = [], resetVersion = 0 }: { atlas: Atlas; frame: ActivityFrame | null; silenced?: number[]; resetVersion?: number }) {
  const signal = useRef(frame);
  const lesion = useRef(silenced);
  const resetView = useRef<(() => void) | null>(null);
  const repaint = useRef<(() => void) | null>(null);
  useEffect(() => { signal.current = frame; lesion.current = silenced; repaint.current?.(); }, [frame, silenced]);
  useEffect(() => { resetView.current?.(); }, [resetVersion]);
  const host = useRef<HTMLDivElement>(null);

  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let disposed = false;
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-3, 3, 2, -2, .01, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.appendChild(renderer.domElement);
    const anatomy = new THREE.Group();
    scene.add(anatomy);
    resetView.current = () => { anatomy.rotation.set(0, 0, 0); camera.zoom = INITIAL_ZOOM; fit(); };
    let geometry: THREE.BufferGeometry | undefined;
    let material: THREE.ShaderMaterial | undefined;
    let size = new THREE.Vector3(5, 2, 1);

    const fit = () => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      const aspect = Math.max(1, width) / Math.max(1, height);
      // Frame the whole atlas once, rather than changing the camera scale as it rotates.
      const radius = size.length() / 2;
      // Keep a generous fixed frame: it should contain the full atlas at reset
      // and after a drag, but never re-fit while the user is rotating it.
      const halfHeight = Math.max(radius, radius / aspect) * 1.18;
      camera.top = halfHeight; camera.bottom = -halfHeight;
      camera.left = -halfHeight * aspect; camera.right = halfHeight * aspect;
      camera.position.set(0, 0, 10);
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    const load = async () => {
      const { positions, groups, ids } = atlas;
      const xyz: number[] = [], bodyIds: number[] = [];
      const bounds = new THREE.Box3();
      for (let i = 0; i < atlas.ids.length; i++) {
        if (groups[i] >= 3) continue;
        const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
        // Native XY projection at reset. A rigid 180-degree X rotation, never axis-wise stretching.
        const point = new THREE.Vector3(x, -y, -z);
        xyz.push(point.x, point.y, point.z);
        bodyIds.push(ids[i]);
        bounds.expandByPoint(point);
      }
      const center = bounds.getCenter(new THREE.Vector3());
      size = bounds.getSize(new THREE.Vector3());
      const scale = 5 / Math.max(size.x, size.y, size.z);
      for (let i = 0; i < xyz.length; i += 3) {
        xyz[i] = (xyz[i] - center.x) * scale;
        xyz[i + 1] = (xyz[i + 1] - center.y) * scale;
        xyz[i + 2] = (xyz[i + 2] - center.z) * scale;
      }
      size.multiplyScalar(scale);
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
      const activity = new Float32Array(bodyIds.length);
      geometry.setAttribute("activity", new THREE.BufferAttribute(activity, 1));
      const silencedFlag = new Float32Array(bodyIds.length);
      geometry.setAttribute("silenced", new THREE.BufferAttribute(silencedFlag, 1));
      material = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false,
        uniforms: { pixelRatio: { value: Math.min(window.devicePixelRatio, 2) } },
        vertexShader: `attribute float activity; attribute float silenced; varying float strength; varying float cut; uniform float pixelRatio;
          void main() { strength = activity; cut = silenced; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (cut > .5 ? 7.0 : 1.15 + strength * 2.35) * pixelRatio; }`,
        fragmentShader: `varying float strength; varying float cut;
          void main() { float r = length(gl_PointCoord - vec2(.5)); if (r > .5) discard;
          if (cut > .5) { gl_FragColor = vec4(.59, .38, .42, r > .34 ? 1. : .55); return; }
          vec3 anatomy = vec3(1., .918, .816);       // #FFEAD0
          vec3 firing = vec3(.969, .435, .557);      // #F76F8E
          vec3 color = mix(anatomy, firing, smoothstep(.06, .7, strength));
          gl_FragColor = vec4(color,(.56+.4*strength)*(1.-smoothstep(.18,.5,r))); }`,
      });
      const paint = () => {
        if (disposed || !geometry) return;
        const values = new Map(signal.current?.values ?? []);
        const cutCells = new Set(lesion.current);
        for (let i = 0; i < bodyIds.length; i++) { activity[i] = values.get(bodyIds[i]) ?? 0; silencedFlag[i] = cutCells.has(bodyIds[i]) ? 1 : 0; }
        geometry.getAttribute("activity").needsUpdate = true;
        geometry.getAttribute("silenced").needsUpdate = true;
        renderer.render(scene, camera);
      };
      repaint.current = paint;
      anatomy.add(new THREE.Points(geometry, material));
      camera.zoom = INITIAL_ZOOM;
      fit();
      paint();
      setState("ready");
    };
    void load().catch(() => { if (!disposed) setState("error"); });
    const observer = new ResizeObserver(fit);
    observer.observe(element);
    fit();
    let held = false, lastX = 0, lastY = 0;
    const down = (event: PointerEvent) => { held = true; lastX = event.clientX; lastY = event.clientY; renderer.domElement.setPointerCapture(event.pointerId); };
    const move = (event: PointerEvent) => {
      if (!held) return;
      anatomy.rotation.y += (event.clientX - lastX) * .006;
      anatomy.rotation.x += (event.clientY - lastY) * .006;
      lastX = event.clientX; lastY = event.clientY;
    };
    const up = () => { held = false; };
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      camera.zoom = THREE.MathUtils.clamp(camera.zoom * Math.exp(-event.deltaY * .001), .45, 4);
      camera.updateProjectionMatrix();
    };
    renderer.domElement.addEventListener("pointerdown", down);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerup", up);
    renderer.domElement.addEventListener("pointercancel", up);
    renderer.domElement.addEventListener("wheel", wheel, { passive: false });
    let frame = 0;
    const animate = () => {
      if (!document.hidden) renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => {
      disposed = true; resetView.current = null; cancelAnimationFrame(frame); repaint.current = null; observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", down); renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up); renderer.domElement.removeEventListener("pointercancel", up);
      renderer.domElement.removeEventListener("wheel", wheel);
      geometry?.dispose(); material?.dispose(); renderer.dispose(); renderer.domElement.remove();
    };
  }, [atlas]);

  return <>
    <div className="brain-legend"><span><i/>Measured anatomy</span><span><i/>Simulated activity [0–1]</span>{silenced.length > 0 && <span className="cut-key"><i/>Silenced cells</span>}</div>
    <div ref={host} className="three-viewport brain-viewport" aria-label="MaleCNS brain soma atlas. Drag to rotate and scroll to zoom.">
      {state !== "ready" && <span className="neural-load" role="status">{state === "error" ? "Atlas unavailable" : <><i className="spinner"/>Loading anatomy…</>}</span>}

    </div>
  </>;
}

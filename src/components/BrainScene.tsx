import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ActivityFrame } from "../lib/replay";
import type { Atlas } from "../lib/atlas";

/** Real anatomy; model values are looked up by body ID, never by spatial proximity. */
export function BrainScene({ atlas, frame }: { atlas: Atlas; frame: ActivityFrame | null }) {
  const signal = useRef(frame);
  const orbit = useRef(true);
  const resetView = useRef<(() => void) | null>(null);
  const [orbiting, setOrbiting] = useState(true);
  const repaint = useRef<(() => void) | null>(null);
  useEffect(() => { signal.current = frame; repaint.current?.(); }, [frame]);
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
    resetView.current = () => { anatomy.rotation.set(0, 0, 0); fit(); };
    let geometry: THREE.BufferGeometry | undefined;
    let material: THREE.ShaderMaterial | undefined;
    let size = new THREE.Vector3(5, 2, 1);

    const fit = () => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      const aspect = Math.max(1, width) / Math.max(1, height);
      const yawRadius = Math.hypot(size.x, size.z) / 2;
      const tiltedHeight = Math.abs(Math.cos(anatomy.rotation.x)) * size.y / 2 + Math.abs(Math.sin(anatomy.rotation.x)) * yawRadius;
      const halfHeight = Math.max(tiltedHeight, yawRadius / aspect) * 1.08;
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
      material = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false,
        uniforms: { pixelRatio: { value: Math.min(window.devicePixelRatio, 2) } },
        vertexShader: `attribute float activity; varying float strength; uniform float pixelRatio;
          void main() { strength = activity; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (0.9 + strength * 2.0) * pixelRatio; }`,
        fragmentShader: `varying float strength;
          void main() { float r = length(gl_PointCoord - vec2(.5)); if (r > .5) discard;
          vec3 color = mix(vec3(.12,.35,.75), vec3(.2,.95,1.), strength);
          color = mix(color,vec3(1.),smoothstep(.6,1.,strength));
          gl_FragColor = vec4(color,(.28+.65*strength)*(1.-smoothstep(.18,.5,r))); }`,
      });
      const paint = () => {
        if (disposed || !geometry) return;
        const values = new Map(signal.current?.values ?? []);
        for (let i = 0; i < bodyIds.length; i++) activity[i] = values.get(bodyIds[i]) ?? 0;
        geometry.getAttribute("activity").needsUpdate = true;
        renderer.render(scene, camera);
      };
      repaint.current = paint;
      anatomy.add(new THREE.Points(geometry, material));
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
      fit();
    };
    const up = () => { held = false; };
    renderer.domElement.addEventListener("pointerdown", down);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerup", up);
    renderer.domElement.addEventListener("pointercancel", up);
    let frame = 0, previous = performance.now();
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const animate = (now: number) => {
      const dt = Math.min(.05,(now - previous) / 1000); previous = now;
      if (orbit.current && !held && !reducedMotion.matches && !document.hidden) anatomy.rotation.y += dt * .12;
      if (!document.hidden) renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => {
      disposed = true; resetView.current = null; cancelAnimationFrame(frame); repaint.current = null; observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", down); renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up); renderer.domElement.removeEventListener("pointercancel", up);
      geometry?.dispose(); material?.dispose(); renderer.dispose(); renderer.domElement.remove();
    };
  }, [atlas]);

  return <>
    <div className="brain-view-controls">
      <button title="Reset to native XY projection with equal axis scale" onClick={() => { orbit.current = false; setOrbiting(false); resetView.current?.(); }}>XY view</button>
      <button aria-pressed={orbiting} onClick={() => { orbit.current = !orbit.current; setOrbiting(orbit.current); }}>Orbit {orbiting ? "on" : "off"}</button>
    </div>
    <div className="brain-legend">Blue: anatomy · cyan/white: supplied values [0, 1]</div>
    <div ref={host} className="three-viewport brain-viewport" aria-label="MaleCNS brain soma atlas">
      {state !== "ready" && <span className="neural-load" role="status">{state === "error" ? "Atlas unavailable" : "Loading anatomy"}</span>}

    </div>
  </>;
}

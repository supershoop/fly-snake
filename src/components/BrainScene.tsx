import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ActivityFrame } from "../lib/replay";
import type { Atlas } from "../lib/atlas";
import type { VisionStatic } from "../lib/live";

const ROLE_COLORS: Record<string, number> = { "object detectors": 0xdfb672, "looming detectors": 0xe07a7a, "relay cells": 0xffffff, steering: 0x84d7ef, "turn away": 0x84d7ef, "giant fiber · escape": 0xe07a7a };
const FULL_RATE_HZ = 150;

/** Real anatomy; model values are looked up by body ID, never by spatial proximity. */
/** `pathway` (optional) overlays the sensory-to-steering circuit on the anatomy: the real cells, lines between them, live rates. */
export function BrainScene({ atlas, frame, pathway, pathwayRates }: { atlas: Atlas; frame: ActivityFrame | null; pathway?: VisionStatic["pathway"] | null; pathwayRates?: Record<string, number> }) {
  const signal = useRef(frame);
  const rates = useRef(pathwayRates);
  const [showPathway, setShowPathway] = useState(true);
  const labels = useRef<HTMLDivElement>(null);
  const orbit = useRef(true);
  const resetView = useRef<(() => void) | null>(null);
  const [orbiting, setOrbiting] = useState(true);
  const repaint = useRef<(() => void) | null>(null);
  useEffect(() => { signal.current = frame; rates.current = pathwayRates; repaint.current?.(); }, [frame, pathwayRates]);
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
    let paintPathway = () => {}, placeLabels = () => {}, disposePathway = () => {};
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
        paintPathway();
        renderer.render(scene, camera);
      };
      repaint.current = paint;
      anatomy.add(new THREE.Points(geometry, material));
      if (pathway && showPathway) {
        const row = new Map(bodyIds.map((id, i) => [id, i]));
        const nodes = pathway.nodes.map(node => {
          const rows = node.bodyIds.map(id => row.get(id)).filter((i): i is number => i !== undefined);
          const centre = new THREE.Vector3();
          rows.forEach(i => centre.add(new THREE.Vector3(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2])));
          return { ...node, rows, centre: centre.divideScalar(Math.max(1, rows.length)) };
        }).filter(node => node.rows.length);
        const byId = new Map(nodes.map(node => [node.id, node]));
        const edges = pathway.edges.filter(([from, to]) => byId.has(from) && byId.has(to));
        const cellRows = nodes.flatMap(node => node.rows.map(i => ({ i, node })));
        const cellGeometry = new THREE.BufferGeometry();
        cellGeometry.setAttribute("position", new THREE.Float32BufferAttribute(cellRows.flatMap(({ i }) => [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]]), 3));
        const cellColors = new Float32Array(cellRows.length * 3);
        cellGeometry.setAttribute("color", new THREE.BufferAttribute(cellColors, 3));
        const cellMaterial = new THREE.PointsMaterial({ size: 5 * Math.min(window.devicePixelRatio, 2), sizeAttenuation: false, vertexColors: true, transparent: true, opacity: .9, depthTest: false });
        const lineGeometry = new THREE.BufferGeometry();
        lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edges.flatMap(([from, to]) => [...byId.get(from)!.centre.toArray(), ...byId.get(to)!.centre.toArray()]), 3));
        const lineColors = new Float32Array(edges.length * 6);
        lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
        const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthTest: false });
        anatomy.add(new THREE.LineSegments(lineGeometry, lineMaterial), new THREE.Points(cellGeometry, cellMaterial));
        const tags = nodes.map(() => { const tag = document.createElement("div"); tag.className = "pathway-label"; labels.current?.appendChild(tag); return tag; });
        const strength = (id: string) => Math.sqrt(Math.min(1, (rates.current?.[id] ?? 0) / FULL_RATE_HZ));
        paintPathway = () => {
          cellRows.forEach(({ node }, c) => new THREE.Color(ROLE_COLORS[node.role] ?? 0xffffff).multiplyScalar(.25 + .75 * strength(node.id)).toArray(cellColors, c * 3));
          edges.forEach(([from, to], e) => {
            const level = .12 + .88 * Math.min(strength(from), Math.max(strength(to), .35 * strength(from)));
            new THREE.Color(0xffffff).multiplyScalar(level).toArray(lineColors, e * 6);
            new THREE.Color(0xffffff).multiplyScalar(level).toArray(lineColors, e * 6 + 3);
          });
          cellGeometry.getAttribute("color").needsUpdate = true;
          lineGeometry.getAttribute("color").needsUpdate = true;
          nodes.forEach((node, n) => {
            const rate = rates.current?.[node.id] ?? 0;
            tags[n].textContent = `${node.label} ${node.side}${rate ? ` · ${Math.round(rate)} Hz` : ""}`;
            tags[n].classList.toggle("firing", rate > 0);
          });
        };
        placeLabels = () => {
          const { width, height } = element.getBoundingClientRect();
          nodes.forEach((node, n) => {
            const p = node.centre.clone().applyMatrix4(anatomy.matrixWorld).project(camera);
            tags[n].style.transform = `translate(${((p.x + 1) / 2) * width}px, ${((1 - p.y) / 2) * height}px)`;
          });
        };
        disposePathway = () => { tags.forEach(tag => tag.remove()); cellGeometry.dispose(); cellMaterial.dispose(); lineGeometry.dispose(); lineMaterial.dispose(); };
      }
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
      if (!document.hidden) { renderer.render(scene, camera); placeLabels(); }
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => {
      disposed = true; resetView.current = null; cancelAnimationFrame(frame); repaint.current = null; observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", down); renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up); renderer.domElement.removeEventListener("pointercancel", up);
      disposePathway(); geometry?.dispose(); material?.dispose(); renderer.dispose(); renderer.domElement.remove();
    };
  }, [atlas, pathway, showPathway]);

  return <>
    <div className="brain-view-controls">
      <button title="Reset to native XY projection with equal axis scale" onClick={() => { orbit.current = false; setOrbiting(false); resetView.current?.(); }}>XY view</button>
      {pathway && <button aria-pressed={showPathway} title="Overlay the food and threat circuits found in the connectome" onClick={() => setShowPathway(!showPathway)}>Pathway {showPathway ? "on" : "off"}</button>}
      <button aria-pressed={orbiting} onClick={() => { orbit.current = !orbit.current; setOrbiting(orbit.current); }}>Orbit {orbiting ? "on" : "off"}</button>
    </div>
    <div className="brain-legend">Blue: anatomy · cyan/white: supplied values [0, 1]</div>
    <div ref={host} className="three-viewport brain-viewport" aria-label="MaleCNS brain soma atlas">
      <div ref={labels} className="pathway-labels" aria-hidden="true"/>
      {state !== "ready" && <span className="neural-load" role="status">{state === "error" ? "Atlas unavailable" : "Loading anatomy"}</span>}

    </div>
  </>;
}

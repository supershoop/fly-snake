import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ActivityFrame } from "../lib/replay";
import type { Atlas } from "../lib/atlas";
import type { VisionStatic } from "../lib/live";

/** Colour follows the pathway. Validated as one palette with the silenced-cell violet (dataviz validator, dark surface: all checks pass). */
export const PATHWAYS: Record<string, { label: string; color: string; hint: string }> = {
  pursuit: { label: "Food pursuit", color: "#c98500", hint: "Object detectors LC10 -> AOTU relay cells -> steering neuron DNa02" },
  escape: { label: "Escape", color: "#d95926", hint: "Looming detectors LC4 and LPLC2 -> giant fiber DNp01" },
  turnaway: { label: "Turn away", color: "#3987e5", hint: "LC4 -> PVLP relay cells -> DNa01 on the opposite side" },
  feeding: { label: "Feeding", color: "#199e70", hint: "Sugar taste neurons -> feeding motor neuron MN9 (fires when the snake eats)" },
  pain: { label: "Pain", color: "#d55181", hint: "Heat sensors -> punishment dopamine neurons PPL1 (fires when the snake dies)" },
};
const SILENCED = "#9085e9";
const FULL_RATE_HZ = 150;

/** Real anatomy; model values are looked up by body ID, never by spatial proximity. */
const INITIAL_ZOOM = 1.28;

/** `pathway` (optional) overlays the sensory-to-steering circuits on the anatomy: the real cells, lines between them, live rates. */
export function BrainScene({ atlas, frame, silenced = [], pathway, pathwayRates, resetVersion = 0 }: { atlas: Atlas; frame: ActivityFrame | null; silenced?: number[]; pathway?: VisionStatic["pathway"] | null; pathwayRates?: Record<string, number>; resetVersion?: number }) {
  const signal = useRef(frame);
  const lesion = useRef(silenced);
  const rates = useRef(pathwayRates);
  const [choice, setChoice] = useState<string>("all");  // "all" | "off" | one pathway
  const resetView = useRef<(() => void) | null>(null);
  const repaint = useRef<(() => void) | null>(null);
  useEffect(() => { signal.current = frame; lesion.current = silenced; rates.current = pathwayRates; repaint.current?.(); }, [frame, silenced, pathwayRates]);
  useEffect(() => { resetView.current?.(); }, [resetVersion]);
  const host = useRef<HTMLDivElement>(null);

  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let disposed = false;
    let paintPathway = () => {}, disposePathway = () => {};
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
        uniforms: { pixelRatio: { value: Math.min(window.devicePixelRatio, 2) }, dim: { value: 1 } },
        vertexShader: `attribute float activity; attribute float silenced; varying float strength; varying float cut; uniform float pixelRatio;
          void main() { strength = activity; cut = silenced; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = (cut > .5 ? 7.0 : 1.15 + strength * 2.35) * pixelRatio; }`,
        fragmentShader: `varying float strength; varying float cut; uniform float dim;
          void main() { float r = length(gl_PointCoord - vec2(.5)); if (r > .5) discard;
          if (cut > .5) { gl_FragColor = vec4(.565, .522, .914, r > .3 ? 1. : .35); return; }  // silenced cell: violet ring, a colour no pathway uses
          vec3 anatomy = vec3(1., .918, .816);       // #FFEAD0
          vec3 firing = vec3(.969, .435, .557);      // #F76F8E
          vec3 color = mix(anatomy, firing, smoothstep(.06, .7, strength));
          gl_FragColor = vec4(color,(.56+.4*strength)*(1.-smoothstep(.18,.5,r))*mix(dim,1.,strength)); }`,
      });
      const paint = () => {
        if (disposed || !geometry) return;
        const values = new Map(signal.current?.values ?? []);
        const cutCells = new Set(lesion.current);
        for (let i = 0; i < bodyIds.length; i++) { activity[i] = values.get(bodyIds[i]) ?? 0; silencedFlag[i] = cutCells.has(bodyIds[i]) ? 1 : 0; }
        geometry.getAttribute("activity").needsUpdate = true;
        geometry.getAttribute("silenced").needsUpdate = true;
        paintPathway();
        renderer.render(scene, camera);
      };
      repaint.current = paint;
      anatomy.add(new THREE.Points(geometry, material));
      const shownGroups = choice === "off" ? [] : choice === "all" ? Object.keys(PATHWAYS) : [choice];
      material.uniforms.dim.value = choice === "all" || choice === "off" ? 1 : .35;  // a single highlighted pathway stands out from a dimmed brain
      if (pathway && shownGroups.length) {
        const row = new Map(bodyIds.map((id, i) => [id, i]));
        const colorOf = (groups: string[]) => new THREE.Color(PATHWAYS[groups.find(g => shownGroups.includes(g)) ?? groups[0]]?.color ?? "#ffffff");
        const nodes = pathway.nodes.filter(node => node.groups.some(g => shownGroups.includes(g))).map(node => {
          const rows = node.bodyIds.map(id => row.get(id)).filter((i): i is number => i !== undefined);
          const centre = new THREE.Vector3();
          rows.forEach(i => centre.add(new THREE.Vector3(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2])));
          return { ...node, rows, centre: centre.divideScalar(Math.max(1, rows.length)), color: colorOf(node.groups) };
        }).filter(node => node.rows.length);
        const byId = new Map(nodes.map(node => [node.id, node]));
        const cellRows = nodes.flatMap(node => node.rows.map(i => ({ i, node })));
        const cellGeometry = new THREE.BufferGeometry();
        cellGeometry.setAttribute("position", new THREE.Float32BufferAttribute(cellRows.flatMap(({ i }) => [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]]), 3));
        const cellColors = new Float32Array(cellRows.length * 3);
        cellGeometry.setAttribute("color", new THREE.BufferAttribute(cellColors, 3));
        const cellMaterial = new THREE.PointsMaterial({ size: 6 * Math.min(window.devicePixelRatio, 2), sizeAttenuation: false, vertexColors: true, transparent: true, opacity: .95, depthTest: false });
        // edges as solid tubes between the two cell groups' centres: WebGL lines are always 1 px wide, tubes are legible
        const up = new THREE.Vector3(0, 1, 0), tubes = new THREE.Group();
        const edges = pathway.edges.filter(([from, to, group]) => byId.has(from) && byId.has(to) && shownGroups.includes(group)).map(([from, to, group]) => {
          const a = byId.get(from)!.centre, b = byId.get(to)!.centre, span = b.clone().sub(a);
          const tubeMaterial = new THREE.MeshBasicMaterial({ color: PATHWAYS[group]?.color ?? "#ffffff", transparent: true, depthTest: false });
          const tube = new THREE.Mesh(new THREE.CylinderGeometry(.014, .014, span.length(), 8, 1, true), tubeMaterial);
          tube.position.copy(a).addScaledVector(span, .5);
          tube.quaternion.setFromUnitVectors(up, span.clone().normalize());
          tubes.add(tube);
          return { from, to, tubeMaterial, tube };
        });
        anatomy.add(tubes, new THREE.Points(cellGeometry, cellMaterial));
        const strength = (id: string) => Math.sqrt(Math.min(1, (rates.current?.[id] ?? 0) / FULL_RATE_HZ));
        paintPathway = () => {
          cellRows.forEach(({ node }, c) => node.color.clone().multiplyScalar(.35 + .65 * strength(node.id)).toArray(cellColors, c * 3));
          cellGeometry.getAttribute("color").needsUpdate = true;
          edges.forEach(({ from, to, tubeMaterial }) => { tubeMaterial.opacity = .22 + .78 * Math.min(strength(from), Math.max(strength(to), .35 * strength(from))); });
        };
        disposePathway = () => { edges.forEach(({ tube, tubeMaterial }) => { tube.geometry.dispose(); tubeMaterial.dispose(); }); cellGeometry.dispose(); cellMaterial.dispose(); };
      }
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
      disposePathway(); geometry?.dispose(); material?.dispose(); renderer.dispose(); renderer.domElement.remove();
    };
  }, [atlas, pathway, choice]);

  return <>
    {pathway && <div className="pathway-picker" role="group" aria-label="Signal paths to show">
      <button aria-pressed={choice === "all"} onClick={() => setChoice("all")}>All paths</button>
      {Object.entries(PATHWAYS).map(([id, item]) => <button key={id} aria-pressed={choice === id} title={item.hint} onClick={() => setChoice(choice === id ? "all" : id)}><i style={{ background: item.color }}/>{item.label}</button>)}
      <button aria-pressed={choice === "off"} onClick={() => setChoice("off")}>Off</button>
    </div>}
    <div className="brain-legend"><span><i/>Measured anatomy</span><span><i/>Simulated activity [0–1]</span>{silenced.length > 0 && <span className="cut-key"><i style={{ borderColor: SILENCED }}/>Silenced cells</span>}</div>
    <div ref={host} className="three-viewport brain-viewport" aria-label="MaleCNS brain soma atlas. Drag to rotate and scroll to zoom.">
      {state !== "ready" && <span className="neural-load" role="status">{state === "error" ? "Atlas unavailable" : <><i className="spinner"/>Loading anatomy…</>}</span>}

    </div>
  </>;
}

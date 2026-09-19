import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { asset } from '../lib/atlas';

type Model = { binary: string; pivots: Record<string, [number, number, number]>; parts: { group: string; material: string; positionByteOffset: number; positionCount: number; indexByteOffset: number; indexCount: number }[] };
/** An anatomical body view. Add a validated motor/physics adapter here if your experiment needs one. */
export function FlyScene() {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const element = host.current!;
    const controller = new AbortController();
    let disposed = false;
    const scene = new THREE.Scene(), modelRoot = new THREE.Group();
    scene.add(modelRoot);
    const camera = new THREE.PerspectiveCamera(35, 1, .001, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.append(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false; controls.enableZoom = false;
    scene.add(new THREE.HemisphereLight(0xffedda, 0x18202a, 3));
    const light = new THREE.DirectionalLight(0xffdfb2, 4); light.position.set(2, 3, 4); scene.add(light);
    const materials: Record<string, THREE.Material> = {};
    const colors: Record<string, number> = { body: 0x9e6834, black: 0x15110e, red: 0xad331f, ocelli: 0xe6b351, 'bristle-brown': 0x281c10, lower: 0xbb8949, brown: 0x52351f };
    for (const [key,color] of Object.entries(colors)) materials[key] = new THREE.MeshStandardMaterial({color,roughness:.65});
    materials.membrane = new THREE.MeshStandardMaterial({color:0xaabbcc,transparent:true,opacity:.36,side:THREE.DoubleSide,depthWrite:false});
    let radius = .3;
    let corners: THREE.Vector3[] = [];
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(Math.max(1,width),Math.max(1,height),false);
      camera.aspect = width / Math.max(1,height);
      const fov = Math.min(camera.fov*Math.PI/180,2*Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));
      camera.position.set(1,.65,1.5).normalize().multiplyScalar(radius/Math.sin(fov/2)*1.1);
      camera.lookAt(0,0,0); camera.updateProjectionMatrix(); controls.update(); draw();
    };
    const draw = () => {
      if (corners.length) {
        camera.zoom = 1; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
        const projected = corners.map(point => point.clone().project(camera));
        const extent = Math.max(...projected.flatMap(point => [Math.abs(point.x), Math.abs(point.y)]));
        camera.zoom = 1 / (extent * 1.12); camera.updateProjectionMatrix();
      }
      renderer.render(scene,camera);
    };
    controls.addEventListener('change',draw);
    void (async () => {
      const get = async (path:string) => { const r = await fetch(asset(`data/flybody/${path}`),{signal:controller.signal}); if(!r.ok) throw Error('Flybody asset unavailable'); return r; };
      const meta = await (await get('model.json')).json() as Model;
      const buffer = await (await get(meta.binary)).arrayBuffer();
      if(disposed) return;
      for(const part of meta.parts) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffer.slice(part.positionByteOffset,part.positionByteOffset+part.positionCount*12)),3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer.slice(part.indexByteOffset,part.indexByteOffset+part.indexCount*4)),1));
        geometry.computeVertexNormals();
        const mesh = new THREE.Mesh(geometry,materials[part.material] ?? materials.body);
        mesh.position.fromArray(meta.pivots[part.group]); modelRoot.add(mesh);
      }
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      modelRoot.position.sub(bounds.getCenter(new THREE.Vector3()));
      radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
      const half = bounds.getSize(new THREE.Vector3()).multiplyScalar(.5);
      corners = [-1,1].flatMap(x => [-1,1].flatMap(y => [-1,1].map(z => new THREE.Vector3(x*half.x,y*half.y,z*half.z))));
      resize();
    })().catch(e => {if(!disposed) setError(String(e));});
    const observer = new ResizeObserver(resize); observer.observe(element); resize();
    return () => {disposed=true;controller.abort();observer.disconnect();controls.dispose();modelRoot.traverse(object=>{if(object instanceof THREE.Mesh)object.geometry.dispose();});Object.values(materials).forEach(m=>m.dispose());renderer.dispose();renderer.domElement.remove();};
  },[]);
  return <div ref={host} className="three-viewport" aria-label="Flybody anatomical surface, drag to rotate">{error&&<p role="alert">{error}</p>}</div>;
}

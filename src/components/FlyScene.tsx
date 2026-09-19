import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { asset } from '../lib/atlas';

export type FlyDirection = 'left' | 'right' | 'up' | 'down';
export type FlyCommand = { direction: FlyDirection; sequence: number };

/** Displays the rigged fly and plays a GLB animation for each game input. */
export function FlyScene({ command }: { command: FlyCommand | null }) {
  const host = useRef<HTMLDivElement>(null);
  const play = useRef<(direction: FlyDirection) => void>(() => {});
  const requestedDirection = useRef<FlyDirection | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!command) return;
    requestedDirection.current = command.direction;
    play.current(command.direction);
  }, [command?.sequence]);

  useEffect(() => {
    const element = host.current!;
    let disposed = false;
    let frame = 0;
    let lastFrame = performance.now();
    let mixer: THREE.AnimationMixer | null = null;
    let radius = 1;

    const scene = new THREE.Scene();
    const modelRoot = new THREE.Group();
    scene.add(modelRoot);
    const camera = new THREE.PerspectiveCamera(35, 1, .001, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.append(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableZoom = false;
    scene.add(new THREE.HemisphereLight(0xffedda, 0x18202a, 3));
    const light = new THREE.DirectionalLight(0xffdfb2, 4);
    light.position.set(2, 3, 4);
    scene.add(light);

    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      camera.aspect = width / Math.max(1, height);
      const fov = Math.min(camera.fov * Math.PI / 180, 2 * Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect));
      camera.position.set(1, .65, 1.5).normalize().multiplyScalar(radius / Math.sin(fov / 2) * 1.2);
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      controls.update();
    };

    const render = (now: number) => {
      mixer?.update(Math.min(.1, (now - lastFrame) / 1000));
      lastFrame = now;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };

    void new GLTFLoader().loadAsync(asset('data/flybody/drosophila.glb')).then(gltf => {
      if (disposed) return;
      modelRoot.add(gltf.scene);
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      modelRoot.position.sub(bounds.getCenter(new THREE.Vector3()));
      radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
      mixer = new THREE.AnimationMixer(gltf.scene);
      const actions = new Map(gltf.animations.map(clip => [clip.name.toLowerCase(), mixer!.clipAction(clip)]));
      const trigger = (direction: FlyDirection) => {
        mixer!.stopAllAction();
        const action = actions.get(direction);
        if (!action) return;
        action.reset();
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = false;
        action.play();
      };
      play.current = trigger;
      if (requestedDirection.current) trigger(requestedDirection.current);
      resize();
    }).catch(loadError => {
      if (!disposed) setError(`Fly animation unavailable: ${String(loadError)}`);
    });

    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    frame = requestAnimationFrame(render);
    return () => {
      disposed = true;
      play.current = () => {};
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      modelRoot.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) material.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div ref={host} className="three-viewport" aria-label="Animated fly at a directional keyboard. Each live model move presses its matching key; drag to rotate.">{error && <p role="alert">{error}</p>}</div>;
}

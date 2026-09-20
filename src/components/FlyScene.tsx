import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { asset } from '../lib/atlas';

export type FlyDirection = 'left' | 'right' | 'up' | 'down';
export type FlyAnimation = FlyDirection | 'idle' | 'win' | 'pain';
export type FlyCommand = { animation: FlyAnimation; sequence: number };

const flyColors: Record<string, number> = {
  body: 0x9e6834,
  black: 0x15110e,
  red: 0xad331f,
  ocelli: 0xe6b351,
  'bristle-brown': 0x281c10,
  lower: 0xbb8949,
  brown: 0x52351f,
};

// This is the front/keyboard-facing view. Keep it separate so future camera
// framing can be tuned without changing the rig or any animation data.
const DEFAULT_CAMERA_DIRECTION = new THREE.Vector3(1, .65, 1.5)
  .applyAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);

/** How many times the pain clip plays when the snake dies. */
const DEATH_SCENE_REPEATS = 2;

/** Displays the rigged fly and plays a GLB animation for each game input. */
export function FlyScene({ command, onDeathSceneLength }: { command: FlyCommand | null; onDeathSceneLength?: (seconds: number) => void }) {
  const reportLength = useRef(onDeathSceneLength);
  reportLength.current = onDeathSceneLength;
  const host = useRef<HTMLDivElement>(null);
  const play = useRef<(animation: FlyAnimation) => void>(() => {});
  const requestedAnimation = useRef<FlyAnimation | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!command) return;
    requestedAnimation.current = command.animation;
    play.current(command.animation);
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
    controls.enableZoom = true;
    controls.zoomSpeed = .8;
    scene.add(new THREE.HemisphereLight(0xffedda, 0x18202a, 3));
    const light = new THREE.DirectionalLight(0xffdfb2, 4);
    light.position.set(2, 3, -4);  // same side as the default camera, so the visible flank is lit
    scene.add(light);

    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      camera.aspect = width / Math.max(1, height);
      const fov = Math.min(camera.fov * Math.PI / 180, 2 * Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect));
      camera.position.copy(DEFAULT_CAMERA_DIRECTION).normalize().multiplyScalar(radius / Math.sin(fov / 2) * .46);
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
      gltf.scene.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          if (!(material instanceof THREE.MeshStandardMaterial)) continue;
          const color = flyColors[material.name];
          if (color !== undefined) {
            material.color.setHex(color);
            material.roughness = .65;
            material.metalness = 0;
          }
          if (material.name === 'membrane') {
            material.color.setHex(0xaabbcc);
            material.transparent = true;
            material.opacity = .36;
            material.depthWrite = false;
          }
          material.needsUpdate = true;
        }
      });
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      modelRoot.position.sub(bounds.getCenter(new THREE.Vector3()));
      radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
      controls.minDistance = radius * .4;
      controls.maxDistance = radius * 5;
      mixer = new THREE.AnimationMixer(gltf.scene);
      const actions = new Map(gltf.animations.map(clip => [clip.name.toLowerCase(), mixer!.clipAction(clip)]));
      const idle = actions.get('idle');
      const playIdle = () => {
        if (!idle) return;
        idle.reset();
        idle.setLoop(THREE.LoopRepeat, Infinity);
        idle.clampWhenFinished = false;
        idle.play();
      };
      const animationClip = (animation: FlyAnimation) => {
        if (animation !== 'win') return animation;
        return ['win', 'winning', 'victory'].find(name => actions.has(name));
      };
      let dying = false;  // the death scene plays out in full: later moves cannot interrupt it
      const trigger = (animation: FlyAnimation) => {
        if (dying && animation !== 'pain') return;
        mixer!.stopAllAction();
        if (animation === 'idle') {
          playIdle();
          return;
        }
        const clip = animationClip(animation);
        const action = clip ? actions.get(clip) : undefined;
        if (!action) {
          playIdle();
          return;
        }
        action.reset();
        dying = animation === 'pain';
        if (dying) action.setLoop(THREE.LoopRepeat, DEATH_SCENE_REPEATS);
        else action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = false;
        action.play();
      };
      const resumeIdle = (event: THREE.AnimationMixerEventMap['finished']) => {
        if (event.action !== idle) {
          dying = false;
          mixer!.stopAllAction();
          playIdle();
        }
      };
      mixer.addEventListener('finished', resumeIdle);
      play.current = trigger;
      const pain = actions.get('pain');
      if (pain) reportLength.current?.(pain.getClip().duration * DEATH_SCENE_REPEATS);
      if (requestedAnimation.current) trigger(requestedAnimation.current);
      else playIdle();
      resize();
      setLoading(false);
    }).catch(loadError => {
      if (!disposed) {
        setError(`Fly animation unavailable: ${String(loadError)}`);
        setLoading(false);
      }
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

  return <div ref={host} className="three-viewport" aria-label="Animated fly at a directional keyboard. Each live model move presses its matching key; drag to rotate and scroll to zoom.">{loading && <div className="scene-loading" role="status"><i className="spinner"/>Loading rigged fly…</div>}{error && <p role="alert">{error}</p>}</div>;
}

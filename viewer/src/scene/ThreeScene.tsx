import { forwardRef, useEffect, useImperativeHandle, useRef, type RefObject } from "react";
import {
  Box3, Color, Fog, Group, PerspectiveCamera, Raycaster, Scene, Vector2,
  Vector3, WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Cards } from "./cards";
import { EdgeField } from "./edges";
import { computeLayout, type Layout } from "./layout";
import { NeuronField } from "./neurons";
import { THEME } from "../lib/theme";
import { unpackF32, type Frame, type Graph } from "../lib/types";

export interface SceneHandle {
  applyGraph(g: Graph): void;
  applyFrame(f: Frame): void;
}

interface Props {
  tipRef: RefObject<HTMLDivElement>;
  onEdgeCount: (n: number) => void;
}

interface SceneState {
  layout: Layout | null;
  edges: EdgeField | null;
  neurons: NeuronField | null;
  cards: Cards | null;
  acts: Map<string, Float32Array> | null;
  hover: string | null;
  dirty: boolean;
  lastLabelPaint: number;
  root: Group;
  bounds: Box3;
}

function fitToView(camera: PerspectiveCamera, controls: OrbitControls, bounds: Box3): void {
  const cz = (bounds.min.z + bounds.max.z) / 2;
  const zr = (bounds.max.z - bounds.min.z) * 0.15;
  const corners: Vector3[] = [];
  for (const x of [bounds.min.x, bounds.max.x])
    for (const y of [bounds.min.y, bounds.max.y])
      for (const z of [cz - zr, cz + zr])
        corners.push(new Vector3(x, y, z));

  const v = new Vector3();
  for (let i = 0; i < 6; i++) {
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
    let worst = 0;
    for (const c of corners) {
      v.copy(c).project(camera);
      worst = Math.max(worst, Math.abs(v.x), Math.abs(v.y));
    }
    if (worst < 1e-6) break;
    const factor = worst / 0.94;
    if (Math.abs(factor - 1) < 0.005) break;
    const off = camera.position.clone().sub(controls.target).multiplyScalar(factor);
    camera.position.copy(controls.target).add(off);
  }
  controls.update();
}

function frameCamera(
  camera: PerspectiveCamera,
  controls: OrbitControls,
  bounds: Box3,
): void {
  const center = bounds.getCenter(new Vector3());
  const size = bounds.getSize(new Vector3());
  controls.target.copy(center);
  const rough = Math.max(size.x, size.y, size.z);
  camera.position.set(center.x + rough * 0.10, center.y + rough * 0.08, center.z + rough);
  fitToView(camera, controls, bounds);
}

function fmt(v: number): string {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  return a >= 1000 || (a < 0.001 && a > 0) ? v.toExponential(2) : v.toFixed(4);
}

export const ThreeScene = forwardRef<SceneHandle, Props>(function ThreeScene(
  { tipRef, onEdgeCount },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const threeRef = useRef<{
    renderer: WebGLRenderer;
    scene: Scene;
    camera: PerspectiveCamera;
    controls: OrbitControls;
    pointer: Vector2;
    state: SceneState;
  } | null>(null);

  // Prop callback'leri ref'e aliyoruz: useImperativeHandle handle'i stabil
  // tutmak icin [] deps kullaniyoruz, ama handle icindeki fonksiyonlar bu
  // ref'lere erisince her zaman guncel prop degerini okurlar.
  const onEdgeCountRef = useRef(onEdgeCount);
  onEdgeCountRef.current = onEdgeCount;
  const tipRefRef = useRef(tipRef);
  tipRefRef.current = tipRef;

  useImperativeHandle(
    ref,
    () => ({
      applyGraph(g: Graph) {
        const t = threeRef.current;
        if (!t) return;
        const { scene, camera, controls, state } = t;

        scene.remove(state.root);
        state.root = new Group();
        state.layout = computeLayout(g);
        state.edges = new EdgeField(g, state.layout);
        state.neurons = new NeuronField(state.layout);
        state.cards = new Cards(state.layout);
        state.root.add(state.edges.object, state.neurons.object, state.cards.object);
        scene.add(state.root);

        state.bounds.setFromObject(state.root);
        frameCamera(camera, controls, state.bounds);
        state.acts = null;
        state.hover = null;
        state.dirty = true;

        onEdgeCountRef.current(state.edges.count);
      },

      applyFrame(f: Frame) {
        const t = threeRef.current;
        if (!t) return;
        const next = new Map<string, Float32Array>();
        for (const [id, b64] of Object.entries(f.act)) next.set(id, unpackF32(b64));
        t.state.acts = next;
        t.state.dirty = true;
      },
    }),
    []
  );

  useEffect(() => {
    const container = containerRef.current!;

    const renderer = new WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    container.appendChild(renderer.domElement);

    const scene = new Scene();
    scene.background = new Color(THEME.bg);
    scene.fog = new Fog(THEME.fog, 60, 190);

    const camera = new PerspectiveCamera(46, innerWidth / innerHeight, 0.1, 900);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.35;

    const raycaster = new Raycaster();
    raycaster.params.Points = { threshold: 0.6 };
    const pointer = new Vector2(-10, -10);

    const state: SceneState = {
      layout: null, edges: null, neurons: null, cards: null,
      acts: null, hover: null, dirty: true, lastLabelPaint: 0,
      root: new Group(), bounds: new Box3(),
    };

    threeRef.current = { renderer, scene, camera, controls, pointer, state };

    // ---------------------------------------------------------- etkileşim
    renderer.domElement.addEventListener("pointermove", (e) => {
      pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
      const tip = tipRefRef.current.current;
      if (tip) {
        tip.style.left = `${e.clientX + 16}px`;
        tip.style.top = `${e.clientY + 16}px`;
      }
    });

    renderer.domElement.addEventListener("pointerdown", () => {
      controls.autoRotate = false;
    });

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "r" || e.key === "R") {
        if (state.layout) frameCamera(camera, controls, state.bounds);
        controls.autoRotate = true;
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    const handleResize = () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
      if (state.layout) fitToView(camera, controls, state.bounds);
    };
    window.addEventListener("resize", handleResize);

    // ---------------------------------------------------------- hover seçimi
    function pick(): void {
      if (!state.neurons || !state.layout) return;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(state.neurons.object, false)[0];
      const key = hit?.index !== undefined ? state.neurons.keys[hit.index] : null;
      if (key === state.hover) return;

      state.hover = key;
      state.dirty = true;

      const tip = tipRefRef.current.current;
      if (!tip) return;
      if (!key) { tip.style.display = "none"; return; }

      const lid = key.slice(0, key.lastIndexOf(":"));
      const i = +key.slice(key.lastIndexOf(":") + 1);
      const pl = state.layout.byId.get(lid);
      if (!pl) return;
      const v = state.acts?.get(lid);
      const name = pl.layer.neuronLabels?.[i];
      const real =
        pl.count < pl.layer.size
          ? Math.round((i * (pl.layer.size - 1)) / Math.max(1, pl.count - 1))
          : i;
      tip.textContent =
        `${pl.layer.label}\n` +
        (name ? `${name}\n` : "") +
        `noron ${real} / ${pl.layer.size}\n` +
        (v && i < v.length ? `aktivasyon ${fmt(v[i])}` : "aktivasyon yok");
      tip.style.display = "block";
    }

    // ---------------------------------------------------------- render döngüsü
    let rafId = 0;
    function tick(now: number): void {
      rafId = requestAnimationFrame(tick);
      controls.update();
      pick();

      if (state.dirty) {
        state.edges?.paint(state.acts, state.hover);
        state.neurons?.paint(state.acts, state.hover);
        state.dirty = false;
        if (now - state.lastLabelPaint > 140) {
          state.cards?.refreshLabels(state.acts);
          state.lastLabelPaint = now;
        }
      }
      renderer.render(scene, camera);
    }
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
      threeRef.current = null;
    };
  }, []);

  return <div ref={containerRef} style={{ position: "fixed", inset: 0 }} />;
});

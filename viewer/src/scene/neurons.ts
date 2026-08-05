import {
  AdditiveBlending, BufferAttribute, BufferGeometry, CanvasTexture,
  Points, ShaderMaterial, Texture,
} from "three";
import { THEME, intensity } from "../lib/theme";
import type { Layout } from "./layout";

/**
 * Butun aglardaki butun noronlar tek bir Points nesnesi.
 *
 * PointsMaterial nokta basina boyut vermeye izin vermedigi icin kucuk bir
 * shader yaziyoruz: boyut ve renk vertex attribute'u olarak geliyor, boylece
 * aktivasyon hem parlakligi hem noktanin buyuklugunu suruyor ve "ates eden
 * noron" gozle aninda secilebiliyor.
 */
export class NeuronField {
  readonly object: Points;
  private colors: Float32Array;
  private sizes: Float32Array;
  private geom: BufferGeometry;
  /** global nokta indisi -> "layerId:index" */
  readonly keys: string[] = [];
  private baseSize: number;

  constructor(layout: Layout) {
    const n = layout.totalPoints;
    const pos = new Float32Array(n * 3);
    this.colors = new Float32Array(n * 3);
    this.sizes = new Float32Array(n);
    this.baseSize = 9;

    let k = 0;
    for (const pl of layout.layers) {
      for (let i = 0; i < pl.count; i++) {
        const p = pl.points[i];
        pos[k * 3] = p.x; pos[k * 3 + 1] = p.y; pos[k * 3 + 2] = p.z;
        this.keys.push(`${pl.layer.id}:${i}`);
        k++;
      }
    }

    this.geom = new BufferGeometry();
    this.geom.setAttribute("position", new BufferAttribute(pos, 3));
    this.geom.setAttribute("color", new BufferAttribute(this.colors, 3));
    this.geom.setAttribute("size", new BufferAttribute(this.sizes, 1));

    this.object = new Points(this.geom, new ShaderMaterial({
      uniforms: { map: { value: glowTexture() } },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform sampler2D map;
        varying vec3 vColor;
        void main() {
          vec4 t = texture2D(map, gl_PointCoord);
          if (t.a < 0.02) discard;
          gl_FragColor = vec4(vColor, t.a);
        }`,
      vertexColors: true,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
    }));
    this.object.frustumCulled = false;
    this.paint(null, null);
  }

  paint(acts: Map<string, Float32Array> | null, hover: string | null): void {
    const layerOf = (key: string) => key.slice(0, key.lastIndexOf(":"));
    const idxOf = (key: string) => +key.slice(key.lastIndexOf(":") + 1);

    for (let k = 0; k < this.keys.length; k++) {
      const key = this.keys[k];
      let a = 0;
      if (acts) {
        const v = acts.get(layerOf(key));
        const i = idxOf(key);
        if (v && i < v.length) a = intensity(v[i]);
      }

      const isHover = key === hover;
      const col = a > 0.55 ? THEME.neuronHot : THEME.neuron;
      const amp = isHover ? 1 : 0.35 + 0.65 * a;

      this.colors[k * 3] = isHover ? THEME.hover.r : col.r * amp;
      this.colors[k * 3 + 1] = isHover ? THEME.hover.g : col.g * amp;
      this.colors[k * 3 + 2] = isHover ? THEME.hover.b : col.b * amp;
      this.sizes[k] = this.baseSize * (isHover ? 2.0 : 0.7 + 0.9 * a);
    }
    (this.geom.getAttribute("color") as BufferAttribute).needsUpdate = true;
    (this.geom.getAttribute("size") as BufferAttribute).needsUpdate = true;
  }
}

/** Yumusak kenarli parlak nokta dokusu (radyal gradyan). */
function glowTexture(): Texture {
  const S = 64;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  grad.addColorStop(0.0, "rgba(255,255,255,1)");
  grad.addColorStop(0.28, "rgba(255,255,255,0.92)");
  grad.addColorStop(0.55, "rgba(255,255,255,0.22)");
  grad.addColorStop(1.0, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  const t = new CanvasTexture(c);
  t.needsUpdate = true;
  return t;
}

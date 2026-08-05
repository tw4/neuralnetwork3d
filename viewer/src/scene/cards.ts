import {
  BufferAttribute, BufferGeometry, CanvasTexture, DoubleSide, Group,
  LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial, PlaneGeometry,
} from "three";
import { THEME, intensity } from "../lib/theme";
import type { Layout, PlacedLayer } from "./layout";

const FONT = '600 %dpx ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const DPR = 3;   // dokuyu 3x cizip kucultmek metni keskin tutar

/**
 * Katman kartlari (baslik / boyut / [aktivasyon]) ve giris ozellik etiketleri.
 *
 * Metin, canvas'a cizilip doku olarak duzleme yapistiriliyor. Sprite yerine
 * dunya-uzayinda duzlem kullaniyoruz: kamera dondugunde kartlarin da sahneyle
 * birlikte egilmesi referanstaki derinlik hissini veriyor; billboard olsalardi
 * her sey kagit gibi duz gorunurdu.
 */
export class Cards {
  readonly object = new Group();
  private labelMeshes = new Map<string, { mesh: Mesh; texts: string[]; }>();

  constructor(layout: Layout) {
    const clear = 1.3;
    layout.layers.forEach((pl, i) => {
      this.object.add(makeCard(pl, clear + (i % 3) * layout.maxHeight * 0.13));
      this.object.add(makeFrame(pl));
      const labels = pl.layer.neuronLabels;
      if (labels && labels.length) {
        const side = pl.layer.kind === "output" ? 1 : -1;
        const mesh = makeLabelColumn(pl, labels, side, null);
        this.object.add(mesh);
        this.labelMeshes.set(pl.layer.id, { mesh, texts: labels });
        (mesh.userData as any).placed = pl;
        (mesh.userData as any).side = side;
      }
    });
  }

  /** Giris/cikis etiketlerini aktivasyona gore yeniden renklendirir. */
  refreshLabels(acts: Map<string, Float32Array> | null): void {
    if (!acts) return;
    for (const [id, rec] of this.labelMeshes) {
      const v = acts.get(id);
      if (!v) continue;
      const ud = rec.mesh.userData as any;
      const tex = drawLabelColumn(rec.texts, ud.side, v);
      const mat = rec.mesh.material as MeshBasicMaterial;
      mat.map?.dispose();
      mat.map = tex;
      mat.needsUpdate = true;
    }
  }
}

// --------------------------------------------------------------- katman karti
function makeCard(pl: PlacedLayer, lift: number): Mesh {
  const L = pl.layer;
  const lines: [string, string, number][] = [[L.label, THEME.card.title, 30]];

  const sub = L.shapeText ? `${L.size}  ${L.shapeText}` : String(L.size);
  lines.push([sub, THEME.card.dim, 24]);
  if (L.activation) lines.push([`[${L.activation}]`, THEME.card.act, 24]);

  const pad = 14 * DPR;
  const widths: number[] = [];
  const probe = document.createElement("canvas").getContext("2d")!;
  let h = pad;
  for (const [text, , size] of lines) {
    probe.font = FONT.replace("%d", String(size * DPR));
    widths.push(probe.measureText(text).width);
    h += size * DPR * 1.35;
  }
  const w = Math.max(...widths) + pad * 2;
  h += pad * 0.4;

  const c = document.createElement("canvas");
  c.width = Math.ceil(w); c.height = Math.ceil(h);
  const g = c.getContext("2d")!;
  g.textBaseline = "top";
  let y = pad * 0.7;
  for (const [text, color, size] of lines) {
    g.font = FONT.replace("%d", String(size * DPR));
    g.fillStyle = color;
    g.fillText(text, pad, y);
    y += size * DPR * 1.35;
  }

  const scale = 0.011;
  const mesh = new Mesh(
    new PlaneGeometry(c.width * scale, c.height * scale),
    new MeshBasicMaterial({
      map: new CanvasTexture(c), transparent: true,
      depthWrite: false, side: DoubleSide,
    })
  );
  mesh.position.set(
    pl.x + (c.width * scale) / 2 - 0.5,
    pl.height / 2 + (c.height * scale) / 2 + lift,
    pl.points[0].z
  );
  return mesh;
}

// ------------------------------------------------------- sutun cercevesi
function makeFrame(pl: PlacedLayer): LineSegments {
  const w = 0.55, h = pl.height / 2 + 0.75, z = pl.points[0].z;
  const x = pl.x;
  const corners = [
    [x - w, -h, z], [x + w, -h, z], [x + w, h, z], [x - w, h, z],
  ];
  const pts: number[] = [];
  for (let i = 0; i < 4; i++) {
    const a = corners[i], b = corners[(i + 1) % 4];
    pts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const geom = new BufferGeometry();
  geom.setAttribute("position", new BufferAttribute(new Float32Array(pts), 3));
  return new LineSegments(geom, new LineBasicMaterial({
    color: 0x5a7ba8, transparent: true, opacity: 0.35, depthWrite: false,
  }));
}

// --------------------------------------------------- giris ozellik etiketleri
function makeLabelColumn(
  pl: PlacedLayer, texts: string[], side: number, acts: Float32Array | null
): Mesh {
  const tex = drawLabelColumn(texts, side, acts);
  const img = tex.image as HTMLCanvasElement;
  const rows = texts.length;
  const scale = rows > 1 ? (pl.height * (rows / (rows - 1))) / img.height : 0.011;
  const w = img.width * scale, h = img.height * scale;
  const mesh = new Mesh(
    new PlaneGeometry(w, h),
    new MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: DoubleSide })
  );
  mesh.position.set(pl.x + side * (w / 2 + 0.7), 0, pl.points[0].z);
  return mesh;
}

function drawLabelColumn(
  texts: string[], side: number, acts: Float32Array | null
): CanvasTexture {
  const size = 24 * DPR;
  const lineH = size * 1.62;
  const probe = document.createElement("canvas").getContext("2d")!;
  probe.font = FONT.replace("%d", String(size));
  const w = Math.max(...texts.map((t) => probe.measureText(t).width)) + 20 * DPR;
  const h = texts.length * lineH;

  const c = document.createElement("canvas");
  c.width = Math.ceil(w); c.height = Math.ceil(h);
  const g = c.getContext("2d")!;
  g.font = FONT.replace("%d", String(size));
  g.textBaseline = "middle";
  g.textAlign = side < 0 ? "right" : "left";
  const x = side < 0 ? w - 8 * DPR : 8 * DPR;

  texts.forEach((t, i) => {
    const a = acts && i < acts.length ? intensity(acts[i]) : 0;
    g.fillStyle = a > 0.55 ? THEME.label.hi : a > 0.2 ? THEME.label.mid : THEME.label.lo;
    g.globalAlpha = 0.45 + 0.55 * a;
    g.fillText(t, x, i * lineH + lineH / 2);
  });

  const tex = new CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

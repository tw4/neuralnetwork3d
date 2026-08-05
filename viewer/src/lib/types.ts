/** Python tarafindaki schema.py'nin birebir karsiligi. */

export type LayerKind =
  | "input" | "dense" | "conv" | "rnn" | "reshape" | "passthrough" | "output";

export interface Layer {
  id: string;
  label: string;
  kind: LayerKind;
  size: number;
  params: number;
  activation?: string;
  shapeText?: string;
  neuronLabels?: string[];
  display: { maxNeurons: number };
}

export interface Edge {
  from: string;
  to: string;
  /** dense: agirlik matrisi var · identity: i->i · reshape: yapisal · recurrent: kendine */
  kind: "dense" | "identity" | "reshape" | "recurrent";
  shape?: [number, number];
  /** base64 float32, satir-oncelikli shape[0]*shape[1] */
  weights?: string;
  weightsCount?: number;
}

export interface Group {
  label: string;
  layers: string[];
  repeat: number;
}

export interface Graph {
  version: number;
  name: string;
  layers: Layer[];
  edges: Edge[];
  groups: Group[];
  framework?: string;
  totalParams?: number;
}

export interface Frame {
  step: number;
  epoch?: number;
  metrics?: Record<string, number>;
  /** katman id -> base64 float32 aktivasyonlar */
  act: Record<string, string>;
}

/** base64 -> Float32Array (little-endian; her yerde oyle uretiyoruz). */
export function unpackF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

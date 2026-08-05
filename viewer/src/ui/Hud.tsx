import type { RefObject } from "react";

export interface StatusInfo {
  kind: "live" | "idle" | "err";
  text: string;
}

export interface MetaInfo {
  name: string;
  layerCount: number;
  edgeCount: number;
  totalParams?: number;
}

function fmt(v: number): string {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  return a >= 1000 || (a < 0.001 && a > 0) ? v.toExponential(2) : v.toFixed(4);
}

interface Props {
  status: StatusInfo;
  meta: MetaInfo | null;
  metrics: Record<string, number>;
  tipRef: RefObject<HTMLDivElement>;
}

export function Hud({ status, meta, metrics, tipRef }: Props) {
  const hasMetrics = Object.keys(metrics).length > 0;

  return (
    <div id="hud">
      <div id="meta" className="panel">
        <div className="name">{meta?.name ?? "nn3d"}</div>
        <div className="sub">
          {meta
            ? `${meta.layerCount} katman · ${meta.edgeCount.toLocaleString("tr")} kenar` +
              (meta.totalParams
                ? ` · ${meta.totalParams.toLocaleString("tr")} parametre`
                : "")
            : "baglaniyor…"}
        </div>
      </div>

      {hasMetrics && (
        <div id="metrics" className="panel">
          {Object.entries(metrics).map(([k, v]) => (
            <div key={k} className="row">
              <span className="k">{k}</span>
              <span className="v">{fmt(v)}</span>
            </div>
          ))}
        </div>
      )}

      <div id="status" className="panel">
        <span
          id="dot"
          className={
            status.kind === "live" ? "live" : status.kind === "err" ? "err" : ""
          }
        />
        <span id="statustext">{status.text}</span>
      </div>

      <div id="help" className="panel">
        <div>
          <kbd>surukle</kbd> dondur · <kbd>tekerlek</kbd> yakinlas
        </div>
        <div>
          <kbd>uzerine gel</kbd> noronu incele · <kbd>R</kbd> sifirla
        </div>
      </div>

      {/* ThreeScene tarafindan imperativ olarak guncellenir */}
      <div id="tip" ref={tipRef} />
    </div>
  );
}

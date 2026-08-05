import { useCallback, useRef, useState } from "react";
import type { Graph, Frame } from "./lib/types";
import { ThreeScene, type SceneHandle } from "./scene/ThreeScene";
import { Hud, type MetaInfo, type StatusInfo } from "./ui/Hud";
import { useSSE } from "./hooks/useSSE";

export function App() {
  const sceneRef = useRef<SceneHandle>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const edgeCountRef = useRef(0);

  const [status, setStatus] = useState<StatusInfo>({ kind: "idle", text: "BAGLANIYOR" });
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number>>({});

  const handleEdgeCount = useCallback((n: number) => {
    edgeCountRef.current = n;
  }, []);

  const handleGraph = useCallback((g: Graph) => {
    // applyGraph is called first; it synchronously calls handleEdgeCount
    // so edgeCountRef.current is ready when we read it below.
    sceneRef.current?.applyGraph(g);
    setMeta({
      name: g.name,
      layerCount: g.layers.length,
      edgeCount: edgeCountRef.current,
      totalParams: g.totalParams,
    });
    setStatus({ kind: "idle", text: "BAGLI · veri bekleniyor" });
  }, []);

  const handleFrame = useCallback((f: Frame) => {
    sceneRef.current?.applyFrame(f);
    if (f.metrics && Object.keys(f.metrics).length) setMetrics(f.metrics);
    const ep = f.epoch !== undefined ? `epoch ${f.epoch} · ` : "";
    setStatus({ kind: "live", text: `CANLI · ${ep}adim ${f.step}` });
  }, []);

  const handleError = useCallback(() => {
    setStatus({ kind: "err", text: "BAGLANTI KOPTU · yeniden deneniyor" });
  }, []);

  useSSE(handleGraph, handleFrame, handleError);

  return (
    <>
      <ThreeScene ref={sceneRef} tipRef={tipRef} onEdgeCount={handleEdgeCount} />
      <Hud status={status} meta={meta} metrics={metrics} tipRef={tipRef} />
    </>
  );
}

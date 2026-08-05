import { useEffect, useRef } from "react";
import type { Frame, Graph } from "../lib/types";

export function useSSE(
  onGraph: (g: Graph) => void,
  onFrame: (f: Frame) => void,
  onError: () => void
) {
  const onGraphRef = useRef(onGraph);
  const onFrameRef = useRef(onFrame);
  const onErrorRef = useRef(onError);
  onGraphRef.current = onGraph;
  onFrameRef.current = onFrame;
  onErrorRef.current = onError;

  useEffect(() => {
    const es = new EventSource("/api/stream");
    es.addEventListener("graph", (e) =>
      onGraphRef.current(JSON.parse((e as MessageEvent).data) as Graph)
    );
    es.addEventListener("frame", (e) =>
      onFrameRef.current(JSON.parse((e as MessageEvent).data) as Frame)
    );
    es.onerror = () => onErrorRef.current();
    return () => es.close();
  }, []);
}

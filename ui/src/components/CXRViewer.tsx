"use client";

import { useRef, useCallback } from "react";
import { imageUrl } from "@/lib/api";
import { useStudyStore } from "@/stores/studyStore";

interface CXRViewerProps {
  imagePath: string;
  priorImagePath?: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function CXRViewer({ imagePath, priorImagePath }: CXRViewerProps) {
  const {
    brightness,
    contrast,
    zoom,
    panX,
    panY,
    setBrightness,
    setContrast,
    setZoom,
    setPan,
    resetView,
  } = useStudyStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom(Math.max(0.5, Math.min(5, zoom * delta)));
    },
    [zoom, setZoom]
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging.current) return;
      const dx = e.clientX - lastPos.current.x;
      const dy = e.clientY - lastPos.current.y;
      lastPos.current = { x: e.clientX, y: e.clientY };
      setPan(panX + dx, panY + dy);
    },
    [panX, panY, setPan]
  );

  const handleMouseUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Image viewport */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden relative bg-black cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          src={imageUrl(imagePath)}
          alt="CXR"
          className="absolute top-1/2 left-1/2 max-w-none select-none"
          draggable={false}
          style={{
            filter: `brightness(${brightness}%) contrast(${contrast}%)`,
            transform: `translate(-50%, -50%) translate(${panX}px, ${panY}px) scale(${zoom})`,
            maxHeight: "100%",
            maxWidth: "100%",
            objectFit: "contain",
          }}
        />
      </div>

      {/* Controls bar */}
      <div className="bg-bg-surface border-t border-separator px-3 py-2">
        <div className="flex items-center gap-3 text-[11px] text-text-secondary">
          <label className="flex items-center gap-1.5">
            <span>Brightness</span>
            <input
              type="range"
              min={20}
              max={300}
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-20 accent-accent"
            />
            <span className="w-8 text-right text-text-tertiary">{brightness}%</span>
          </label>
          <label className="flex items-center gap-1.5">
            <span>Contrast</span>
            <input
              type="range"
              min={20}
              max={300}
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              className="w-20 accent-accent"
            />
            <span className="w-8 text-right text-text-tertiary">{contrast}%</span>
          </label>
          <button
            onClick={resetView}
            className="px-2 py-0.5 rounded bg-bg-elevated text-text-primary hover:text-white transition-colors text-[10px]"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

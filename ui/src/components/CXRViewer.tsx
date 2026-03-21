"use client";

import { useRef, useCallback, useMemo } from "react";
import { imageUrl } from "@/lib/api";
import { useStudyStore } from "@/stores/studyStore";

interface CXRViewerProps {
  imagePath: string;
  priorImagePath?: string;
  priorStudyDate?: string;
  lateralImagePath?: string;
}

export default function CXRViewer({
  imagePath,
  priorImagePath,
  priorStudyDate,
  lateralImagePath,
}: CXRViewerProps) {
  const {
    brightness,
    contrast,
    zoom,
    panX,
    panY,
    activeImageTab,
    setBrightness,
    setContrast,
    setZoom,
    setPan,
    resetView,
    setActiveImageTab,
  } = useStudyStore();

  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  // Build available tabs
  const tabs = useMemo(() => {
    const t: Array<{ key: "current" | "prior" | "lateral"; label: string }> = [
      { key: "current", label: "Current" },
    ];
    if (priorImagePath) {
      const dateLabel = priorStudyDate ? ` (${priorStudyDate})` : "";
      t.push({ key: "prior", label: `Prior${dateLabel}` });
    }
    if (lateralImagePath) {
      t.push({ key: "lateral", label: "Lateral" });
    }
    return t;
  }, [priorImagePath, priorStudyDate, lateralImagePath]);

  const hasTabs = tabs.length > 1;

  // Resolve which image to display
  const displayPath = useMemo(() => {
    if (activeImageTab === "prior" && priorImagePath) return priorImagePath;
    if (activeImageTab === "lateral" && lateralImagePath) return lateralImagePath;
    return imagePath;
  }, [activeImageTab, imagePath, priorImagePath, lateralImagePath]);

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
      {/* Image tab bar — only shown when multiple views available */}
      {hasTabs && (
        <div className="bg-bg-surface border-b border-separator px-3 flex items-center gap-0">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => {
                setActiveImageTab(tab.key);
                resetView();
              }}
              className={`px-3 py-1.5 text-xs font-medium transition-colors border-b-2 ${
                activeImageTab === tab.key
                  ? "text-white border-white"
                  : "text-white/50 border-transparent hover:text-white/80"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}

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
          src={imageUrl(displayPath)}
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
        <div className="flex items-center gap-3 text-xs text-text-secondary">
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

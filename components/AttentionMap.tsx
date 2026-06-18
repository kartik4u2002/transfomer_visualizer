"use client";

import React, { useState } from "react";
import MatrixHeatmap from "./MatrixHeatmap";

interface AttentionMapProps {
  tokens: string[];
  attentionWeights: number[][];
}

export default function AttentionMap({
  tokens,
  attentionWeights,
}: AttentionMapProps) {
  const [activeQueryIdx, setActiveQueryIdx] = useState<number | null>(null);
  const [activeKeyIdx, setActiveKeyIdx] = useState<number | null>(null);

  const numTokens = tokens.length;
  const svgWidth = 500;
  const svgHeight = 160;
  const tokenY = svgHeight - 30; // Y-position of token centers in SVG

  // Get X coordinate for a token index
  const getTokenX = (idx: number) => {
    return (idx + 0.5) * (svgWidth / numTokens);
  };

  // Color palette for Query tokens to distinguish arcs
  const queryColors = [
    "stroke-indigo-500",   // token 0
    "stroke-emerald-500",  // token 1
    "stroke-violet-500",   // token 2
    "stroke-amber-500",    // token 3
    "stroke-cyan-500",     // token 4
    "stroke-rose-500",     // token 5
  ];

  const queryHexColors = [
    "#6366f1", // indigo-500
    "#10b981", // emerald-500
    "#8b5cf6", // violet-500
    "#f59e0b", // amber-500
    "#06b6d4", // cyan-500
    "#f43f5e", // rose-500
  ];

  return (
    <div className="flex flex-col gap-6 w-full">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        
        {/* SVG Arcs Visualization */}
        <div className="flex flex-col bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/80 rounded-xl p-5 shadow-sm backdrop-blur-sm">
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Attention Flow Arcs
            </h4>
            <p className="text-xs text-slate-400 dark:text-slate-500">
              Arcs flow from Query token (top curve start) to Key token (endpoint). Hover a token to filter.
            </p>
          </div>

          {/* SVG Canvas */}
          <div className="relative w-full aspect-[5/2] lg:aspect-auto lg:h-[180px] bg-white dark:bg-slate-950/40 rounded-lg border border-slate-100 dark:border-slate-800/60 p-2 flex flex-col justify-between">
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full h-full overflow-visible"
            >
              {/* Define Arrow Markers */}
              <defs>
                {queryHexColors.map((color, qIdx) => (
                  <marker
                    key={qIdx}
                    id={`arrow-${qIdx}`}
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill={color} opacity="0.8" />
                  </marker>
                ))}
              </defs>

              {/* Render Arcs */}
              {attentionWeights.map((row, qIdx) =>
                row.map((weight, kIdx) => {
                  if (weight < 0.01) return null; // Skip tiny weights

                  const xQ = getTokenX(qIdx);
                  const xK = getTokenX(kIdx);
                  const color = queryHexColors[qIdx % queryHexColors.length];
                  
                  // Highlight status
                  const isHovered = activeQueryIdx === qIdx || activeKeyIdx === kIdx;
                  const isAnyHovered = activeQueryIdx !== null || activeKeyIdx !== null;
                  const opacity = isAnyHovered
                    ? isHovered
                      ? weight * 0.95 + 0.05
                      : weight * 0.1 // Dim non-hovered arcs
                    : weight * 0.8 + 0.1;

                  const strokeWidth = isAnyHovered && isHovered
                    ? weight * 10 + 1.5
                    : weight * 7 + 0.5;

                  // Self-attention (loop)
                  if (qIdx === kIdx) {
                    const r = 16 + weight * 8;
                    const pathD = `M ${xQ - 4} ${tokenY - 12} A ${r} ${r} 0 1 1 ${xQ + 4} ${tokenY - 12}`;
                    return (
                      <path
                        key={`arc-${qIdx}-${kIdx}`}
                        d={pathD}
                        fill="none"
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeOpacity={opacity}
                        className="transition-all duration-200"
                      />
                    );
                  }

                  // Cross-attention curve (quadratic bezier)
                  const midX = (xQ + xK) / 2;
                  const dist = Math.abs(qIdx - kIdx);
                  const curveHeight = 35 + dist * 25; // Arc height scales with distance
                  const pathD = `M ${xQ} ${tokenY - 12} Q ${midX} ${tokenY - 12 - curveHeight} ${xK} ${tokenY - 12}`;

                  return (
                    <path
                      key={`arc-${qIdx}-${kIdx}`}
                      d={pathD}
                      fill="none"
                      stroke={color}
                      strokeWidth={strokeWidth}
                      strokeOpacity={opacity}
                      markerEnd={`url(#arrow-${qIdx % queryHexColors.length})`}
                      className="transition-all duration-200"
                    />
                  );
                })
              )}

              {/* Render Labels inside SVG */}
              {tokens.map((token, idx) => {
                const x = getTokenX(idx);
                const isHovered = activeQueryIdx === idx || activeKeyIdx === idx;
                return (
                  <g key={idx} className="cursor-pointer">
                    {/* Circle Node */}
                    <circle
                      cx={x}
                      cy={tokenY - 8}
                      r="6"
                      className={`transition-all duration-200 ${
                        isHovered
                          ? "fill-slate-800 dark:fill-slate-100 r-8"
                          : "fill-slate-300 dark:fill-slate-700"
                      }`}
                    />
                  </g>
                );
              })}
            </svg>

            {/* Tokens Bar */}
            <div className="flex justify-around w-full mt-1 border-t border-slate-100 dark:border-slate-800/40 pt-2">
              {tokens.map((token, idx) => {
                const colorBorder = queryHexColors[idx % queryHexColors.length];
                const isActive = activeQueryIdx === idx || activeKeyIdx === idx;

                return (
                  <button
                    key={idx}
                    className={`px-3 py-1 rounded-full text-xs font-mono font-medium border transition-all duration-150 ${
                      isActive
                        ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950 border-slate-800 dark:border-white shadow-md scale-105"
                        : "bg-slate-50 text-slate-700 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 border-slate-200/60 dark:border-slate-800/60"
                    }`}
                    style={{
                      borderLeftWidth: "4px",
                      borderLeftColor: colorBorder,
                    }}
                    onMouseEnter={() => {
                      setActiveQueryIdx(idx);
                    }}
                    onMouseLeave={() => {
                      setActiveQueryIdx(null);
                    }}
                  >
                    {token}
                  </button>
                );
              })}
            </div>
          </div>
          
          {/* Hover Details Panel */}
          <div className="mt-3 text-[11px] text-slate-500 dark:text-slate-400 min-h-[18px] text-center font-mono bg-slate-100/50 dark:bg-slate-900/20 py-1 rounded">
            {activeQueryIdx !== null ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: queryHexColors[activeQueryIdx] }} />
                <span>
                  Query: <strong>&ldquo;{tokens[activeQueryIdx]}&rdquo;</strong> sends attention to:
                </span>
                <span className="text-slate-600 dark:text-slate-300">
                  {attentionWeights[activeQueryIdx]
                    .map((w, i) => `${tokens[i]}: ${(w * 100).toFixed(0)}%`)
                    .join(", ")}
                </span>
              </span>
            ) : (
              <span>Hover a token above to trace its Attention Query distribution</span>
            )}
          </div>
        </div>

        {/* Matrix Heatmap Grid */}
        <div className="flex flex-col justify-between">
          <MatrixHeatmap
            matrix={attentionWeights}
            rowLabels={tokens.map((t) => `${t} (Q)`)}
            colLabels={tokens.map((t) => `${t} (K)`)}
            title="Attention Weights Matrix"
            subtitle="Result of softmax(Q · K^T / sqrt(d_k)). Rows sum to 1.0."
          />
        </div>

      </div>
    </div>
  );
}

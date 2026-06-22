"use client";

import React, { useState } from "react";

interface CrossAttentionMapProps {
  attentionWeights: number[][];
  decoderTokens: string[];
  encoderTokens: string[];
  title?: string;
  subtitle?: string;
}

export default function CrossAttentionMap({
  attentionWeights,
  decoderTokens,
  encoderTokens,
  title = "Encoder-Decoder Cross-Attention Weights",
  subtitle = "How each decoder token (query) attends to each encoder token (key)",
}: CrossAttentionMapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    r: number;
    c: number;
    val: number;
  } | null>(null);

  const numRows = attentionWeights.length;
  const numCols = attentionWeights[0]?.length || 0;

  // Since attention weights are probabilities, the max absolute value is at most 1.0.
  // We can scale cell opacity by the weight value itself (since weights are in [0, 1]).
  const getCellColor = (val: number) => {
    // Red-500 equivalent alpha for positive weights
    return `rgba(239, 68, 68, ${val * 0.85})`;
  };

  const getTextColor = (val: number) => {
    if (val > 0.45) {
      return "text-white font-semibold";
    }
    return "text-slate-800 dark:text-slate-200";
  };

  return (
    <div className="flex flex-col w-full bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/80 rounded-xl p-4 shadow-sm backdrop-blur-sm">
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
          {title}
        </h4>
        <p className="text-xs text-slate-400 dark:text-slate-500">
          {subtitle}
        </p>
      </div>

      <div className="relative overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-800">
        <table className="border-collapse mx-auto select-none">
          <thead>
            <tr>
              {/* Empty top-left label */}
              <th className="p-2 text-xs font-semibold text-slate-400 dark:text-slate-500 text-right min-w-[80px]">
                Decoder \ Encoder
              </th>
              {encoderTokens.map((token, cIdx) => (
                <th
                  key={cIdx}
                  className="p-2 text-center text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono tracking-wider min-w-[64px]"
                >
                  {token}
                  <span className="block text-[9px] text-slate-400 dark:text-slate-500 font-sans font-normal">
                    (K_{cIdx})
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attentionWeights.map((row, rIdx) => (
              <tr key={rIdx}>
                <td className="p-2 text-right text-xs font-semibold text-slate-700 dark:text-slate-355 font-sans pr-4 whitespace-nowrap min-w-[80px]">
                  <span className="font-bold">{decoderTokens[rIdx]}</span>
                  <span className="block text-[9px] text-slate-400 dark:text-slate-500 font-mono font-normal">
                    (Q_{rIdx})
                  </span>
                </td>
                {row.map((val, cIdx) => {
                  const cellBg = getCellColor(val);
                  return (
                    <td
                      key={cIdx}
                      className="p-0 border border-slate-200/60 dark:border-slate-800/60"
                      onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, val })}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      <div
                        className={`h-10 flex items-center justify-center transition-all duration-150 relative cursor-crosshair text-xs font-mono px-3 min-w-[64px] ${getTextColor(
                          val
                        )}`}
                        style={{ backgroundColor: cellBg }}
                      >
                        {val.toFixed(2)}

                        {/* Tooltip on hover */}
                        {hoveredCell?.r === rIdx && hoveredCell?.c === cIdx && (
                          <div className="absolute z-30 bottom-full mb-2 bg-slate-950 text-slate-50 text-[10px] py-1.5 px-2.5 rounded-md shadow-xl border border-slate-800 pointer-events-none whitespace-nowrap flex flex-col items-center">
                            <span className="font-semibold text-slate-300">
                              Weight: {(val * 100).toFixed(1)}%
                            </span>
                            <span className="text-[9px] text-slate-400 mt-0.5">
                              Decoder &ldquo;{decoderTokens[rIdx]}&rdquo; &rarr; Encoder &ldquo;{encoderTokens[cIdx]}&rdquo;
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Heatmap Legend */}
      <div className="flex items-center justify-between mt-3 px-1 border-t border-slate-200/40 dark:border-slate-800/40 pt-3 text-[10px] text-slate-400 dark:text-slate-500">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950" />
          <span>No Attention (0.00)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-red-500/80" />
          <span>Full Attention (1.00)</span>
        </div>
      </div>
    </div>
  );
}

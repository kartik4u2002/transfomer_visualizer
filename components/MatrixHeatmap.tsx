"use client";

import React, { useState } from "react";

interface MatrixHeatmapProps {
  matrix: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  title?: string;
  subtitle?: string;
}

export default function MatrixHeatmap({
  matrix,
  rowLabels,
  colLabels,
  title,
  subtitle,
}: MatrixHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    r: number;
    c: number;
    val: number;
  } | null>(null);

  const numRows = matrix.length;
  const numCols = matrix[0]?.length || 0;

  // Find max absolute value for centering color scale at 0
  const flatVals = matrix.flat();
  const maxAbs = Math.max(...flatVals.map(Math.abs), 0.0001);

  // Helper to determine if we should display numbers inside the cells
  // If we have many columns (e.g. FFN hidden dim = 16), text will overflow.
  const showTextInCell = numCols <= 6;

  // Format value to 4 decimal places
  const formatVal = (v: number) => v.toFixed(4);

  // Get color for a value based on diverging scale (Red for +, Blue for -)
  const getCellColor = (val: number) => {
    const ratio = Math.min(Math.abs(val) / maxAbs, 1);
    // Coral red for positive: HSL 350 (red)
    // Sky blue for negative: HSL 215 (blue)
    if (val >= 0) {
      return `rgba(239, 68, 68, ${ratio * 0.85})`; // tailwind red-500 equivalent alpha
    } else {
      return `rgba(59, 130, 246, ${ratio * 0.85})`; // tailwind blue-500 equivalent alpha
    }
  };

  // Get contrasting text color based on cell intensity
  const getTextColor = (val: number) => {
    const ratio = Math.abs(val) / maxAbs;
    if (ratio > 0.45) {
      return "text-white font-semibold";
    }
    return "text-slate-800 dark:text-slate-200";
  };

  return (
    <div className="flex flex-col w-full bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/80 rounded-xl p-4 shadow-sm backdrop-blur-sm">
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              {title}
            </h4>
          )}
          {subtitle && (
            <p className="text-xs text-slate-400 dark:text-slate-500">
              {subtitle}
            </p>
          )}
        </div>
      )}

      <div className="relative overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-800">
        <table className="border-collapse mx-auto select-none">
          <thead>
            <tr>
              {/* Top-left empty header cell */}
              {rowLabels && <th className="p-2 text-xs font-semibold text-slate-400 dark:text-slate-500 text-right min-w-[60px]" />}
              {colLabels &&
                colLabels.map((lbl, cIdx) => (
                  <th
                    key={cIdx}
                    className="p-2 text-center text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono tracking-wider min-w-[50px]"
                  >
                    {lbl}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => (
              <tr key={rIdx}>
                {rowLabels && (
                  <td className="p-2 text-right text-xs font-semibold text-slate-700 dark:text-slate-300 font-sans pr-4 whitespace-nowrap min-w-[60px]">
                    {rowLabels[rIdx]}
                  </td>
                )}
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
                        className={`h-10 flex items-center justify-center transition-all duration-150 relative cursor-crosshair text-xs font-mono px-3 ${
                          showTextInCell ? "min-w-[64px]" : "w-8 min-w-[32px]"
                        } ${getTextColor(val)}`}
                        style={{ backgroundColor: cellBg || undefined }}
                      >
                        {showTextInCell ? (
                          val === 0 ? "0" : val.toFixed(2)
                        ) : (
                          <span className="w-1.5 h-1.5 rounded-full bg-current opacity-30" />
                        )}

                        {/* Tooltip on hover */}
                        {hoveredCell?.r === rIdx && hoveredCell?.c === cIdx && (
                          <div className="absolute z-30 bottom-full mb-2 bg-slate-950 text-slate-50 text-[10px] py-1.5 px-2.5 rounded-md shadow-xl border border-slate-800 pointer-events-none whitespace-nowrap flex flex-col items-center">
                            <span className="font-semibold text-slate-300">
                              Value: {formatVal(val)}
                            </span>
                            {rowLabels && colLabels && (
                              <span className="text-[9px] text-slate-400 mt-0.5">
                                {rowLabels[rIdx]} → {colLabels[cIdx]}
                              </span>
                            )}
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
          <span className="w-2.5 h-2.5 rounded bg-blue-500/80" />
          <span>Negative (<span className="font-mono">-{formatVal(maxAbs)}</span>)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950" />
          <span>Zero (<span className="font-mono">0.00</span>)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded bg-red-500/80" />
          <span>Positive (<span className="font-mono">+{formatVal(maxAbs)}</span>)</span>
        </div>
      </div>
    </div>
  );
}

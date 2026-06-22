"use client";

import React, { useState } from "react";
import { Lock, Unlock } from "lucide-react";

interface MaskGridProps {
  tokens: string[];
}

export default function MaskGrid({ tokens }: MaskGridProps) {
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number } | null>(null);

  return (
    <div className="flex flex-col w-full bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/80 rounded-xl p-4 shadow-sm backdrop-blur-sm">
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
          Causal Look-Ahead Mask Grid
        </h4>
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Causal attention prevents positions from attending to future tokens (where key index &gt; query index).
        </p>
      </div>

      <div className="overflow-x-auto pb-2">
        <table className="border-collapse mx-auto select-none">
          <thead>
            <tr>
              <th className="p-2 text-xs font-semibold text-slate-400 dark:text-slate-500 text-right min-w-[70px]">
                Query \ Key
              </th>
              {tokens.map((token, cIdx) => (
                <th
                  key={cIdx}
                  className="p-2 text-center text-xs font-semibold text-slate-500 dark:text-slate-400 font-mono tracking-wider min-w-[70px]"
                >
                  {token}
                  <span className="block text-[9px] text-slate-400 dark:text-slate-500 font-normal font-sans">
                    j = {cIdx}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((qToken, rIdx) => (
              <tr key={rIdx}>
                <td className="p-2 text-right text-xs font-semibold text-slate-700 dark:text-slate-350 font-sans pr-4 whitespace-nowrap min-w-[70px]">
                  <span className="font-semibold">{qToken}</span>
                  <span className="block text-[9px] text-slate-400 dark:text-slate-500 font-mono font-normal">
                    i = {rIdx}
                  </span>
                </td>
                {tokens.map((kToken, cIdx) => {
                  const isMasked = cIdx > rIdx;
                  const isHovered = hoveredCell?.r === rIdx && hoveredCell?.c === cIdx;

                  return (
                    <td
                      key={cIdx}
                      className="p-1 border border-slate-200/60 dark:border-slate-800/60"
                      onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx })}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      <div
                        className={`h-12 w-16 md:w-20 rounded-lg flex flex-col items-center justify-center transition-all duration-200 relative cursor-help text-xs font-semibold ${
                          isMasked
                            ? "bg-slate-200/40 text-slate-450 dark:bg-slate-950/60 dark:text-slate-500 border border-dashed border-slate-300 dark:border-slate-800"
                            : "bg-emerald-550/10 text-emerald-600 dark:bg-emerald-550/20 dark:text-emerald-400 border border-emerald-500/20 dark:border-emerald-500/30"
                        } ${isHovered ? "scale-105 shadow-md z-10" : ""}`}
                      >
                        {isMasked ? (
                          <>
                            <Lock className="w-3.5 h-3.5 text-slate-450 dark:text-slate-500 mb-0.5" />
                            <span className="text-[9px] font-mono font-bold uppercase tracking-wider">
                              -&infin;
                            </span>
                          </>
                        ) : (
                          <>
                            <Unlock className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400 mb-0.5" />
                            <span className="text-[9px] font-mono font-bold uppercase tracking-wider">
                              Pass
                            </span>
                          </>
                        )}

                        {/* Tooltip */}
                        {isHovered && (
                          <div className="absolute z-30 bottom-full mb-2 bg-slate-900 text-slate-50 text-[10px] py-2 px-3 rounded-lg shadow-xl border border-slate-800 pointer-events-none w-48 text-left leading-normal">
                            <div className="font-bold text-slate-200">
                              Query: &ldquo;{qToken}&rdquo; (i={rIdx})
                            </div>
                            <div className="text-slate-300">
                              Key: &ldquo;{kToken}&rdquo; (j={cIdx})
                            </div>
                            <div className="mt-1.5 border-t border-slate-800 pt-1">
                              {isMasked ? (
                                <span className="text-red-400 font-medium">
                                  MASKED: Cannot attend to future position {cIdx}. Attention score set to -&infin; (0% weight).
                                </span>
                              ) : (
                                <span className="text-emerald-400 font-medium">
                                  ALLOWED: Can attend to past/present position {cIdx}.
                                </span>
                              )}
                            </div>
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

      <div className="flex items-center gap-3 mt-3 px-1 border-t border-slate-200/40 dark:border-slate-800/40 pt-3 text-[10px] text-slate-400 dark:text-slate-500">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-emerald-500/10 border border-emerald-500/20" />
          <span>Allowed (Key index &le; Query index)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-slate-200/40 dark:bg-slate-950/60 border border-dashed border-slate-300 dark:border-slate-800" />
          <span>Masked (Key index &gt; Query index)</span>
        </div>
      </div>
    </div>
  );
}

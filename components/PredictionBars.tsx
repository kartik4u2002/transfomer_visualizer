"use client";

import React from "react";
import { Sparkles, ArrowRight } from "lucide-react";

interface PredictionBarsProps {
  probabilities: number[][];
  vocab: string[];
}

export default function PredictionBars({
  probabilities,
  vocab,
}: PredictionBarsProps) {
  const lastIndex = probabilities.length - 1;
  const lastProbabilities = probabilities[lastIndex] || [];

  // Combine words and their corresponding probabilities
  const predictionItems = lastProbabilities.map((prob, idx) => ({
    word: vocab[idx] || `[unknown_${idx}]`,
    prob,
    isArgmax: false, // will set below
  }));

  // Sort descending by probability
  predictionItems.sort((a, b) => b.prob - a.prob);

  // Mark argmax (the highest probability item)
  if (predictionItems.length > 0) {
    predictionItems[0].isArgmax = true;
  }

  // Get top 5 predictions
  const topPredictions = predictionItems.slice(0, 5);

  return (
    <div className="flex flex-col w-full bg-slate-50/50 dark:bg-slate-900/40 border border-slate-100 dark:border-slate-800/80 rounded-xl p-5 shadow-sm backdrop-blur-sm">
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-indigo-500 animate-pulse" />
          <span>Next Token Prediction Probability Distribution</span>
        </h4>
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Softmax probabilities for the final decoder sequence position (pos {lastIndex}).
        </p>
      </div>

      <div className="flex flex-col gap-3">
        {topPredictions.map((item, idx) => {
          const percentage = (item.prob * 100).toFixed(1);
          
          return (
            <div
              key={idx}
              className={`flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-xl border transition-all duration-200 ${
                item.isArgmax
                  ? "bg-indigo-500/10 dark:bg-indigo-500/20 border-indigo-500/30 dark:border-indigo-500/40 shadow-xs"
                  : "bg-white/40 dark:bg-slate-950/20 border-slate-200/50 dark:border-slate-800/40"
              }`}
            >
              {/* Token Name and argmax badge */}
              <div className="flex items-center gap-2 min-w-[120px]">
                <span
                  className={`font-mono text-xs px-2.5 py-1 rounded-md ${
                    item.isArgmax
                      ? "bg-indigo-650 text-white font-bold"
                      : "bg-slate-100 dark:bg-slate-850 text-slate-700 dark:text-slate-300"
                  }`}
                >
                  &ldquo;{item.word}&rdquo;
                </span>
                {item.isArgmax && (
                  <span className="text-[9px] uppercase font-extrabold tracking-wider bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 px-1.5 py-0.5 rounded flex items-center gap-0.5 animate-pulse">
                    <ArrowRight className="w-2.5 h-2.5" />
                    Argmax Pick
                  </span>
                )}
              </div>

              {/* Progress Bar Container */}
              <div className="flex-1 flex items-center gap-3">
                <div className="flex-1 h-3.5 bg-slate-100 dark:bg-slate-850 rounded-full overflow-hidden border border-slate-200/40 dark:border-slate-800/40">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ease-out ${
                      item.isArgmax
                        ? "bg-indigo-600 dark:bg-indigo-500"
                        : "bg-slate-450 dark:bg-slate-650"
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span
                  className={`text-xs font-mono font-bold min-w-[45px] text-right ${
                    item.isArgmax
                      ? "text-indigo-600 dark:text-indigo-400"
                      : "text-slate-600 dark:text-slate-400"
                  }`}
                >
                  {percentage}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {topPredictions.length > 0 && (
        <div className="mt-4 p-3 bg-emerald-500/5 border border-emerald-500/10 rounded-xl flex items-center gap-3 text-xs text-slate-600 dark:text-slate-400">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shrink-0 font-bold font-mono">
            &alpha;
          </div>
          <div>
            The model selects the token with the highest probability (argmax). In this execution step, the predicted next token is <strong className="text-slate-800 dark:text-slate-100 font-mono font-bold">&ldquo;{topPredictions[0].word}&rdquo;</strong> with <strong className="text-indigo-600 dark:text-indigo-400">{(topPredictions[0].prob * 100).toFixed(1)}%</strong> confidence.
          </div>
        </div>
      )}
    </div>
  );
}

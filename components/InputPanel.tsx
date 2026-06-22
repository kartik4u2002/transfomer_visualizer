"use client";

import React, { useState, useEffect } from "react";
import { Sparkles, AlertTriangle } from "lucide-react";

interface InputPanelProps {
  initialSentence: string;
  onSubmit: (sentence: string) => void;
  title?: string;
  description?: string;
  presets?: string[];
  placeholder?: string;
}

const DEFAULT_PRESETS = [
  "cat rat mat",
  "the dog ran",
  "i like math",
  "neural networks learn",
  "vectors model text",
];

export default function InputPanel({
  initialSentence,
  onSubmit,
  title = "Interactive Input Sentence",
  description = "Type a short sentence (up to 6 words) to trace its math live.",
  presets = DEFAULT_PRESETS,
  placeholder = "e.g. cat rat mat",
}: InputPanelProps) {
  const [text, setText] = useState(initialSentence);
  const [error, setError] = useState<string | null>(null);

  // Validate the text and calculate word count
  const getWordCount = (val: string) => {
    return val
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0).length;
  };

  useEffect(() => {
    const wordCount = getWordCount(text);
    if (wordCount > 6) {
      setError("Maximum sentence length is 6 words for matrix legibility.");
    } else if (wordCount === 0) {
      setError("Sentence cannot be empty.");
    } else {
      setError(null);
    }
  }, [text]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const wordCount = getWordCount(text);
    if (wordCount > 0 && wordCount <= 6) {
      onSubmit(text.trim());
    }
  };

  const handlePresetClick = (preset: string) => {
    setText(preset);
    onSubmit(preset);
  };

  return (
    <div className="w-full bg-white dark:bg-slate-950/20 border border-slate-200/60 dark:border-slate-800/60 rounded-2xl p-5 md:p-6 shadow-sm">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* Label & Description */}
        <div>
          <label
            htmlFor={`sentence-input-${title.replace(/\s+/g, "-").toLowerCase()}`}
            className="text-sm font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5"
          >
            <Sparkles className="w-4 h-4 text-indigo-500" />
            <span>{title}</span>
          </label>
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
            {description}
          </p>
        </div>

        {/* Input Bar */}
        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-1">
            <input
              id={`sentence-input-${title.replace(/\s+/g, "-").toLowerCase()}`}
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className={`w-full px-4 py-3 rounded-xl border text-sm font-medium transition-all bg-slate-50 dark:bg-slate-900/60 text-slate-800 dark:text-slate-100 placeholder-slate-400 focus:outline-hidden focus:ring-2 ${
                error
                  ? "border-red-300 dark:border-red-900/40 focus:ring-red-500/20"
                  : "border-slate-200 dark:border-slate-800 focus:ring-indigo-500/20 focus:border-indigo-500"
              }`}
              placeholder={placeholder}
            />
          </div>

          <button
            type="submit"
            disabled={!!error}
            className="px-6 py-3 text-sm font-bold rounded-xl bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shrink-0 shadow-sm"
          >
            Visualize Math
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400 font-medium bg-red-50 dark:bg-red-950/20 px-3 py-2 rounded-lg border border-red-100 dark:border-red-900/20">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Presets List */}
        <div className="flex flex-wrap items-center gap-2 mt-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mr-1 select-none">
            Presets:
          </span>
          {presets.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => handlePresetClick(preset)}
              className={`text-xs px-3 py-1 rounded-lg border transition-all duration-150 cursor-pointer ${
                text.trim().toLowerCase() === preset.toLowerCase()
                  ? "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-900/50 font-semibold"
                  : "bg-slate-50/50 hover:bg-slate-100/80 dark:bg-slate-900/40 dark:hover:bg-slate-800/80 text-slate-500 dark:text-slate-400 border-slate-200/60 dark:border-slate-800/60"
              }`}
            >
              {preset}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
}

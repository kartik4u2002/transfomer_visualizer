"use client";

import React, { useEffect, useRef } from "react";
import katex from "katex";

interface FormulaBlockProps {
  formula: string;
  block?: boolean;
  caption?: string;
}

export default function FormulaBlock({
  formula,
  block = true,
  caption,
}: FormulaBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(formula, containerRef.current, {
          displayMode: block,
          throwOnError: false,
        });
      } catch (error) {
        console.error("KaTeX rendering error:", error);
      }
    }
  }, [formula, block]);

  return (
    <div className="my-2 select-none">
      <div
        ref={containerRef}
        className="overflow-x-auto py-2 text-slate-800 dark:text-slate-100 scrollbar-thin scrollbar-thumb-slate-200 dark:scrollbar-thumb-slate-800"
      />
      {caption && (
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-medium text-center">
          {caption}
        </p>
      )}
    </div>
  );
}

"use client";

import React from "react";
import { ChevronLeft, ChevronRight, Check } from "lucide-react";

interface StepperProps {
  currentStep: number;
  totalSteps: number;
  onChange: (step: number) => void;
  stageNames: string[];
}

export default function Stepper({
  currentStep,
  totalSteps,
  onChange,
  stageNames,
}: StepperProps) {
  const handlePrev = () => {
    if (currentStep > 0) onChange(currentStep - 1);
  };

  const handleNext = () => {
    if (currentStep < totalSteps - 1) onChange(currentStep + 1);
  };

  return (
    <div className="w-full flex flex-col gap-4 bg-white dark:bg-slate-950/20 border border-slate-200/60 dark:border-slate-800/60 rounded-2xl p-4 md:p-6 shadow-sm">
      {/* Top Navigation Row */}
      <div className="flex items-center justify-between w-full gap-2">
        <button
          onClick={handlePrev}
          disabled={currentStep === 0}
          className="flex items-center gap-1.5 px-3 py-2 text-xs md:text-sm font-semibold rounded-xl transition-all duration-150 border border-slate-200/80 dark:border-slate-800/80 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Prev</span>
        </button>

        {/* Current Stage Title Indicator (mobile-focused) */}
        <div className="flex flex-col text-center">
          <span className="text-[10px] uppercase font-bold tracking-widest text-slate-400 dark:text-slate-500">
            Stage {currentStep + 1} of {totalSteps}
          </span>
          <span className="text-sm md:text-base font-bold text-slate-800 dark:text-slate-100 transition-all">
            {stageNames[currentStep]}
          </span>
        </div>

        <button
          onClick={handleNext}
          disabled={currentStep === totalSteps - 1}
          className="flex items-center gap-1.5 px-3 py-2 text-xs md:text-sm font-semibold rounded-xl transition-all duration-150 border border-indigo-500/25 dark:border-indigo-500/40 bg-indigo-600 dark:bg-indigo-700 text-white hover:bg-indigo-700 dark:hover:bg-indigo-600 disabled:opacity-40 disabled:bg-slate-200 dark:disabled:bg-slate-800 disabled:border-transparent dark:disabled:border-transparent disabled:text-slate-400 dark:disabled:text-slate-600 disabled:cursor-not-allowed cursor-pointer shadow-sm shadow-indigo-500/10"
        >
          <span>Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* Desktop Stepper Bar */}
      <div className="hidden md:flex items-center justify-between w-full relative px-2 py-4">
        {/* Gray Connector Line Background */}
        <div className="absolute top-[28px] left-[5%] right-[5%] h-0.5 bg-slate-100 dark:bg-slate-800 -z-10" />
        
        {/* Colored Progress Line Overlay */}
        <div
          className="absolute top-[28px] left-[5%] h-0.5 bg-indigo-500 dark:bg-indigo-600 transition-all duration-300 -z-10"
          style={{ width: `${(currentStep / (totalSteps - 1)) * 90}%` }}
        />

        {stageNames.map((name, idx) => {
          const isActive = idx === currentStep;
          const isCompleted = idx < currentStep;

          return (
            <button
              key={idx}
              onClick={() => onChange(idx)}
              className="flex flex-col items-center group cursor-pointer w-[11%]"
            >
              {/* Dot Node */}
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center border-2 text-[11px] font-bold font-mono transition-all duration-200 ${
                  isActive
                    ? "bg-indigo-600 border-indigo-600 text-white scale-110 shadow-lg shadow-indigo-500/25 ring-4 ring-indigo-500/15"
                    : isCompleted
                    ? "bg-emerald-500 border-emerald-500 text-white"
                    : "bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-800 text-slate-400 dark:text-slate-500 group-hover:border-slate-400 dark:group-hover:border-slate-600"
                }`}
              >
                {isCompleted ? <Check className="w-3.5 h-3.5 stroke-[3]" /> : idx + 1}
              </div>

              {/* Stage Name */}
              <span
                className={`mt-2 text-[10px] font-medium tracking-tight text-center max-w-[80px] leading-tight select-none transition-colors duration-200 ${
                  isActive
                    ? "text-indigo-600 dark:text-indigo-400 font-bold"
                    : isCompleted
                    ? "text-slate-600 dark:text-slate-400"
                    : "text-slate-400 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-400"
                }`}
              >
                {name}
              </span>
            </button>
          );
        })}
      </div>

      {/* Mobile Dot Indicator Row */}
      <div className="flex md:hidden justify-center gap-1.5 py-1">
        {stageNames.map((_, idx) => {
          const isActive = idx === currentStep;
          const isCompleted = idx < currentStep;
          return (
            <button
              key={idx}
              onClick={() => onChange(idx)}
              className={`h-2 rounded-full transition-all duration-200 cursor-pointer ${
                isActive
                  ? "w-6 bg-indigo-600 dark:bg-indigo-500"
                  : isCompleted
                  ? "w-2.5 bg-emerald-500"
                  : "w-2 bg-slate-200 dark:bg-slate-800"
              }`}
              title={stageNames[idx]}
            />
          );
        })}
      </div>
    </div>
  );
}

"use client";

import React, { useState, useMemo } from "react";
import InputPanel from "../components/InputPanel";
import Stepper from "../components/Stepper";
import StageCard from "../components/StageCard";
import { runTransformerPipeline } from "../lib/transformer";
import { DEFAULT_SENTENCE, DEFAULT_D_MODEL, DEFAULT_HIDDEN_DIM } from "../data/sampleSentence";
import { HelpCircle, GraduationCap } from "lucide-react";

const STAGE_NAMES = [
  "1. Tokenizer",
  "2. Embedding Layer",
  "3. Positional Encoding",
  "4. Self-Attention",
  "5. Add & Norm #1",
  "6. Feed-Forward Network",
  "7. Add & Norm #2",
  "8. Final Embeddings",
];

export default function Home() {
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [currentStep, setCurrentStep] = useState(0);

  // Compute pipeline results memoized on sentence change
  const pipelineResult = useMemo(() => {
    return runTransformerPipeline(sentence, DEFAULT_D_MODEL, DEFAULT_HIDDEN_DIM);
  }, [sentence]);

  const handleSentenceChange = (newSentence: string) => {
    setSentence(newSentence);
    // Reset back to tokenizer stage when a new sentence is entered
    setCurrentStep(0);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#030712] text-slate-900 dark:text-slate-50 flex flex-col transition-colors duration-150">
      {/* Navbar Header */}
      <header className="border-b border-slate-200/60 dark:border-slate-800/60 bg-white/70 dark:bg-[#030712]/70 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-600/15">
              <GraduationCap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-bold text-sm sm:text-base leading-tight tracking-tight text-slate-850 dark:text-slate-50">
                Transformer Visualizer
              </h1>
              <p className="text-[10px] text-slate-400 dark:text-slate-500 font-medium">
                Step-by-Step Mathematical Walkthrough
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden md:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 border border-emerald-200/50 dark:border-emerald-900/35">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Client-Side Computation
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 flex flex-col gap-8 w-full">
        {/* Banner Section */}
        <div className="bg-radial from-indigo-500/5 to-indigo-500/0 dark:from-indigo-500/10 dark:to-transparent border border-slate-200/50 dark:border-slate-800/60 rounded-3xl p-6 md:p-8 flex flex-col gap-4 text-center md:text-left items-center md:items-start shadow-xs">
          <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white max-w-2xl">
            Demystifying the math inside a Transformer Encoder
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-3xl leading-relaxed">
            Ever get lost in the matrix shapes during self-attention? This tool visualizes, with exact numbers, every single dot product, softmax row, mean, variance, and ReLU activation inside one encoder block. Type a short sentence below to calculate values live.
          </p>
        </div>

        {/* Legend Panel & Dimension Notation */}
        <section className="bg-white dark:bg-slate-950/20 border border-slate-200/60 dark:border-slate-800/60 rounded-2xl p-5 shadow-xs flex flex-col gap-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-1.5 select-none">
            <HelpCircle className="w-4 h-4 text-slate-400" />
            <span>Mathematical Tensor Notation Legend</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            <div className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800/40 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">batch_size (B) = 1</span>
              <p className="text-slate-400 dark:text-slate-500">
                Number of sequences processed simultaneously. We evaluate a single sequence here.
              </p>
            </div>
            <div className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800/40 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-indigo-600 dark:text-indigo-400 font-mono">seq_len (T) = {pipelineResult.tokens.length}</span>
              <p className="text-slate-400 dark:text-slate-500">
                Sequence length, equal to the number of word tokens in the input sentence.
              </p>
            </div>
            <div className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800/40 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">d_model = {DEFAULT_D_MODEL}</span>
              <p className="text-slate-400 dark:text-slate-500">
                Vector representation size for each token. Kept small for easy visualization.
              </p>
            </div>
            <div className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800/40 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">hidden_dim = {DEFAULT_HIDDEN_DIM}</span>
              <p className="text-slate-400 dark:text-slate-500">
                Intermediate FFN projection size. Projects features outward to hidden activation space.
              </p>
            </div>
          </div>
        </section>

        {/* Input Configuration Panel */}
        <InputPanel initialSentence={sentence} onSubmit={handleSentenceChange} />

        {/* Stepper Navigation */}
        <Stepper
          currentStep={currentStep}
          totalSteps={STAGE_NAMES.length}
          onChange={setCurrentStep}
          stageNames={STAGE_NAMES}
        />

        {/* Interactive Active Stage Display */}
        <div className="transition-all duration-300">
          <StageCard step={currentStep} data={pipelineResult} />
        </div>
      </main>

      {/* Footer Branding */}
      <footer className="border-t border-slate-200/60 dark:border-slate-800/60 bg-white dark:bg-[#030712] py-8 text-center text-xs text-slate-400 dark:text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-650 dark:text-slate-450">Transformer Visualizer</span>
            <span>&bull;</span>
            <span>Educational sandbox</span>
          </div>
          <div className="text-[10px]">
            Designed for teachers and students explaining matrix operations. Centered diverging heatmaps, auto-RNG seeds.
          </div>
        </div>
      </footer>
    </div>
  );
}

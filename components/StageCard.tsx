"use client";

import React, { useState } from "react";
import FormulaBlock from "./FormulaBlock";
import MatrixHeatmap from "./MatrixHeatmap";
import AttentionMap from "./AttentionMap";
import { TransformerPipelineResult } from "../lib/transformer";
import { Layers, GitCommit, Database, Eye } from "lucide-react";

interface StageCardProps {
  step: number;
  data: TransformerPipelineResult;
}

export default function StageCard({ step, data }: StageCardProps) {
  // Tabs for stage sub-matrices
  const [attnTab, setAttnTab] = useState<"visualize" | "qkv" | "weights">("visualize");
  const [ffnTab, setFfnTab] = useState<"activations" | "output" | "weights">("activations");

  const { tokens, vocab, tokenIds } = data;

  // Render left explanation panel
  const renderLeftPanel = () => {
    switch (step) {
      case 0: // Tokenizer
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 1
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Tokenizer
              </h3>
            </div>
            <FormulaBlock
              formula="\text{Tokenizer}(S) = [ID_1, ID_2, \dots, ID_T]"
              caption="Mapping a sentence string S to integer Token IDs"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              Tokenization splits the raw input sentence into discrete word units (tokens) by whitespace, converts them to lowercase, and assigns a unique integer index based on a dynamically built vocabulary list.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Sequence Length (<code className="font-mono text-indigo-500">seq_len</code>): {tokens.length} tokens</li>
                <li>Vocab Size (<code className="font-mono text-indigo-500">vocab_size</code>): {vocab.length} unique words</li>
              </ul>
            </div>
          </div>
        );
      case 1: // Embedding
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 2
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Embedding Layer
              </h3>
            </div>
            <FormulaBlock
              formula="X_{embed} = \text{Lookup}(\text{Token IDs}, W_{embed})"
              caption="Looking up embedding vectors for each token ID"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              Word embeddings convert discrete integer IDs into continuous vector representations. Each token ID points to a row in the embedding weights matrix <code className="font-mono text-indigo-500">W_embed</code> of size <code className="font-mono text-indigo-500">(vocab_size, d_model)</code>. The looked-up matrix represents the initial vector space of our input sentence.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Weight Matrix <code className="font-mono text-indigo-500">W_embed</code>: ({vocab.length} &times; {data.W_embed[0].length})</li>
                <li>Embeddings Output <code className="font-mono text-indigo-500">X_embed</code>: ({tokens.length} &times; {data.embeddings[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 2: // Positional Encoding
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 3
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Positional Encoding
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ Z &= X_{embed} + PE \end{aligned}"
              caption="Computing sinusoids & adding them to word embeddings"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              Transformers process all tokens simultaneously (no recurrence or convolutions). To capture the sequence order, we add a fixed positional encoding matrix <code className="font-mono text-indigo-500">PE</code> composed of sinusoidal signals. This ensures words like &ldquo;rat&rdquo; at position 1 and &ldquo;rat&rdquo; at position 5 have different vectors, encoding order without expanding vector dimensions.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Positional Encoding <code className="font-mono text-indigo-500">PE</code>: ({tokens.length} &times; {data.pe[0].length})</li>
                <li>Encoded Matrix <code className="font-mono text-indigo-500">Z</code>: ({tokens.length} &times; {data.Z[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 3: // Self-Attention
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 4
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Self-Attention
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} Q = Z \cdot W_q, \ &K = Z \cdot W_k, \ V = Z \cdot W_v \\ \text{Scaled} &= \frac{Q \cdot K^T}{\sqrt{d_k}} \\ \text{Weights} &= \text{softmax}(\text{Scaled}) \\ \text{Output} &= \text{Weights} \cdot V \end{aligned}"
              caption="Scales Q, K dot-product and aggregates V"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              Self-attention determines the contextual relation between tokens. 
              <strong> Queries (Q)</strong> search for information, <strong> Keys (K)</strong> advertise the token's information, and <strong> Values (V)</strong> contain the actual content. 
              We calculate Query-Key dot products, scale by <code className="font-mono text-indigo-500">{"1/\\sqrt{d_k}"}</code> (here <code className="font-mono text-indigo-500">{"1/\\sqrt{4} = 0.5"}</code>) to stabilize gradients, and apply a row-wise Softmax to create probability weights. We multiply these weights by <code className="font-mono text-indigo-500">V</code> to compute a weighted contextual representation.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Query, Key, Value (<code className="font-mono text-indigo-500">Q, K, V</code>): ({tokens.length} &times; {data.Q[0].length})</li>
                <li>Attention Weights: ({tokens.length} &times; {tokens.length})</li>
              </ul>
            </div>
          </div>
        );
      case 4: // Add & Norm 1
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 5
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Add & Norm #1
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} X_{res1} &= Z + \text{AttentionOutput} \\ Z_{norm1} &= \text{LayerNorm}(X_{res1}) \end{aligned}"
              caption="Residual connection and Layer Normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add a skip (residual) connection by adding the pre-attention inputs <code className="font-mono text-indigo-500">Z</code> to the attention outputs. This alleviates vanishing gradients during training. 
              We then apply Layer Normalization. For each row (token), we calculate the mean <code className="font-mono text-indigo-500">&mu;</code> and variance <code className="font-mono text-indigo-500">&sigma;^2</code> across the <code className="font-mono text-indigo-500">d_model</code> dimensions and normalize:
            </p>
            <FormulaBlock formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}" block={false} />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res1</code>: ({tokens.length} &times; {data.add1[0].length})</li>
                <li>Normalized output <code className="font-mono text-indigo-500">Z_norm1</code>: ({tokens.length} &times; {data.norm1[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 5: // Feed Forward Network
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 6
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Feed-Forward Network
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} H_{raw} &= X \cdot W_1 + b_1 \\ H_{activated} &= \max(0, H_{raw}) \\ \text{FFN\_Output} &= H_{activated} \cdot W_2 + b_2 \end{aligned}"
              caption="Two-layer neural network with ReLU activation"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              The Feed-Forward Network (FFN) consists of two linear layers with a ReLU activation in between. It is applied to each token vector independently (no communication between tokens occurs in this stage—that is restricted to the attention stage). 
              The intermediate layer projects vectors to a higher-dimensional space (<code className="font-mono text-indigo-500">hidden_dim = 16</code>) to introduce non-linear mapping capacity before projecting back to <code className="font-mono text-indigo-500">d_model = 4</code>.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Hidden Matrix <code className="font-mono text-indigo-500">H_activated</code>: ({tokens.length} &times; {data.W1[0].length})</li>
                <li>FFN Output: ({tokens.length} &times; {data.W2[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 6: // Add & Norm 2
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 7
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Add & Norm #2
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} X_{res2} &= Z_{norm1} + \text{FFN\_Output} \\ \text{Output} &= \text{LayerNorm}(X_{res2}) \end{aligned}"
              caption="Second residual skip connection and layer normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add the inputs of the FFN block (<code className="font-mono text-indigo-500">Z_norm1</code>) to the outputs of the FFN block. This second residual path is then normalized using LayerNorm to yield the final output vectors of our encoder block.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res2</code>: ({tokens.length} &times; {data.add2[0].length})</li>
                <li>Final Output Matrix: ({tokens.length} &times; {data.norm2[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 7: // Final Output
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 8
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Final Contextual Embeddings
              </h3>
            </div>
            <FormulaBlock
              formula="\text{Encoder Output} = Z_{norm2}"
              caption="The contextualized matrix representation"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              This is the final contextual embedding matrix produced by the Transformer Encoder Block. Every row vector represents a word in the input sequence, but unlike the static embeddings of Stage 2, these vectors have incorporated surrounding structural information (context) through self-attention and non-linear mappings. 
              These representations can now be passed to subsequent encoder stages or decoder networks.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Output Shape:</strong>
              <p className="font-mono mt-1 text-slate-600 dark:text-slate-300">
                ({tokens.length} &times; {data.norm2[0].length}) [seq_len, d_model]
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // Render right matrix visualization panel
  const renderRightPanel = () => {
    switch (step) {
      case 0: // Tokenizer
        return (
          <div className="flex flex-col gap-5">
            {/* Vocab List */}
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Vocabulary List (Vocab Lookup Table)
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {vocab.map((v, i) => (
                  <div
                    key={v}
                    className="flex justify-between items-center bg-slate-50 dark:bg-slate-900 px-3 py-2 rounded-lg border border-slate-200/40 dark:border-slate-800/40 font-mono text-xs text-slate-700 dark:text-slate-300"
                  >
                    <span>&ldquo;{v}&rdquo;</span>
                    <span className="font-bold text-indigo-500">{i}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Token IDs Map */}
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Token mapping (Sentence &rarr; ID sequence)
              </h4>
              <div className="flex flex-wrap items-center gap-2">
                {tokens.map((token, idx) => (
                  <div key={idx} className="flex items-center">
                    <div className="flex flex-col items-center bg-slate-50 dark:bg-slate-900 px-4 py-2 border border-slate-200/60 dark:border-slate-800/60 rounded-xl shadow-xs">
                      <span className="text-[10px] text-slate-400 dark:text-slate-500">
                        pos {idx}
                      </span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {token}
                      </span>
                      <span className="font-bold text-indigo-500 font-mono text-xs mt-1 bg-indigo-50 dark:bg-indigo-950/50 px-2 py-0.5 rounded">
                        {tokenIds[idx]}
                      </span>
                    </div>
                    {idx < tokens.length - 1 && (
                      <span className="mx-1.5 text-slate-300 dark:text-slate-700 font-bold">&rarr;</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 1D Matrix Heatmap of IDs */}
            <MatrixHeatmap
              matrix={[tokenIds]}
              rowLabels={["Token ID"]}
              colLabels={tokens}
              title="Token IDs Vector"
              subtitle="The numerical matrix input shape: (1, seq_len)"
            />
          </div>
        );
      case 1: // Embedding
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={data.W_embed}
              rowLabels={vocab.map((v, idx) => `ID ${idx} [${v}]`)}
              colLabels={Array.from({ length: data.W_embed[0].length }, (_, i) => `d_${i}`)}
              title="Embedding Weight Matrix (W_embed)"
              subtitle="Weight values generated with seed 42. Shape: (vocab_size, d_model)"
            />
            <MatrixHeatmap
              matrix={data.embeddings}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.embeddings[0].length }, (_, i) => `d_${i}`)}
              title="Looked Up Token Embeddings (X_embed)"
              subtitle="Values grabbed from matching rows of W_embed. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 2: // Positional Encoding
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={data.pe}
              rowLabels={tokens.map((t, i) => `${t} (pos ${i})`)}
              colLabels={Array.from({ length: data.pe[0].length }, (_, i) => `d_${i}`)}
              title="Positional Encoding Matrix (PE)"
              subtitle="Calculated sine & cosine constants. Shape: (seq_len, d_model)"
            />
            <MatrixHeatmap
              matrix={data.Z}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.Z[0].length }, (_, i) => `d_${i}`)}
              title="Encoded Embeddings Input (Z = X_embed + PE)"
              subtitle="Combined vectors containing both semantic and index context. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 3: // Self-Attention
        return (
          <div className="flex flex-col gap-4">
            {/* Tabs Selector */}
            <div className="flex border-b border-slate-200 dark:border-slate-800">
              <button
                onClick={() => setAttnTab("visualize")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  attnTab === "visualize"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Attention Maps</span>
              </button>
              <button
                onClick={() => setAttnTab("qkv")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  attnTab === "qkv"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>Q, K, V Projections</span>
              </button>
              <button
                onClick={() => setAttnTab("weights")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  attnTab === "weights"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Wq, Wk, Wv Weights</span>
              </button>
            </div>

            <div className="mt-2">
              {attnTab === "visualize" && (
                <AttentionMap tokens={tokens} attentionWeights={data.attentionWeights} />
              )}
              {attnTab === "qkv" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <MatrixHeatmap
                      matrix={data.Q}
                      rowLabels={tokens}
                      colLabels={["q0", "q1", "q2", "q3"]}
                      title="Queries (Q = Z * Wq)"
                    />
                    <MatrixHeatmap
                      matrix={data.K}
                      rowLabels={tokens}
                      colLabels={["k0", "k1", "k2", "k3"]}
                      title="Keys (K = Z * Wk)"
                    />
                    <MatrixHeatmap
                      matrix={data.V}
                      rowLabels={tokens}
                      colLabels={["v0", "v1", "v2", "v3"]}
                      title="Values (V = Z * Wv)"
                    />
                  </div>
                  <MatrixHeatmap
                    matrix={data.attentionOutput}
                    rowLabels={tokens}
                    colLabels={["d0", "d1", "d2", "d3"]}
                    title="Attention Output (Weights * V)"
                    subtitle="Weighted sum of V. Shape: (seq_len, d_model)"
                  />
                </div>
              )}
              {attnTab === "weights" && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MatrixHeatmap
                    matrix={data.Wq}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["q0", "q1", "q2", "q3"]}
                    title="Wq Matrix (seed 101)"
                  />
                  <MatrixHeatmap
                    matrix={data.Wk}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["k0", "k1", "k2", "k3"]}
                    title="Wk Matrix (seed 102)"
                  />
                  <MatrixHeatmap
                    matrix={data.Wv}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["v0", "v1", "v2", "v3"]}
                    title="Wv Matrix (seed 103)"
                  />
                </div>
              )}
            </div>
          </div>
        );
      case 4: // Add & Norm 1
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={data.add1}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.add1[0].length }, (_, i) => `d_${i}`)}
              title="Residual Addition (Z + AttentionOutput)"
              subtitle="Shape: (seq_len, d_model)"
            />

            {/* LayerNorm detailed calculation showing means and variances */}
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Layer Normalization Calculations (Per Row)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                      <th className="py-2 pr-4 font-sans font-semibold">Token</th>
                      <th className="py-2 pr-4 font-semibold text-center">Mean (&mu;)</th>
                      <th className="py-2 pr-4 font-semibold text-center">Variance (&sigma;<sup>2</sup>)</th>
                      <th className="py-2 font-semibold">Normalized Vector</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                    {tokens.map((token, idx) => (
                      <tr key={idx} className="text-slate-700 dark:text-slate-300">
                        <td className="py-2.5 pr-4 font-sans font-bold text-slate-800 dark:text-slate-200">{token}</td>
                        <td className="py-2.5 pr-4 text-center">{data.means1[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{data.variances1[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{data.norm1[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={data.norm1}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.norm1[0].length }, (_, i) => `d_${i}`)}
              title="Layer Normalization Output (Z_norm1)"
              subtitle="Output scaled to mean=0, variance=1. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 5: // Feed Forward
        return (
          <div className="flex flex-col gap-4">
            {/* Tabs Selector */}
            <div className="flex border-b border-slate-200 dark:border-slate-800">
              <button
                onClick={() => setFfnTab("activations")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  ffnTab === "activations"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <GitCommit className="w-3.5 h-3.5" />
                <span>Hidden Activations</span>
              </button>
              <button
                onClick={() => setFfnTab("output")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  ffnTab === "output"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>FFN Output</span>
              </button>
              <button
                onClick={() => setFfnTab("weights")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  ffnTab === "weights"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>Weights & Biases</span>
              </button>
            </div>

            <div className="mt-2">
              {ffnTab === "activations" && (
                <div className="flex flex-col gap-6">
                  <MatrixHeatmap
                    matrix={data.H_raw}
                    rowLabels={tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Pre-Activation Hidden Matrix (H_raw = Z_norm1 * W1 + b1)"
                    subtitle="Calculated dot-products. Value shown on hover. Shape: (seq_len, hidden_dim)"
                  />
                  <MatrixHeatmap
                    matrix={data.H_activated}
                    rowLabels={tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Intermediate Hidden Activation Matrix (H_activated = ReLU(H_raw))"
                    subtitle="Values below 0 are zeroed out (ReLU). Shape: (seq_len, hidden_dim)"
                  />
                </div>
              )}
              {ffnTab === "output" && (
                <MatrixHeatmap
                  matrix={data.ffnOutput}
                  rowLabels={tokens}
                  colLabels={Array.from({ length: 4 }, (_, i) => `d_${i}`)}
                  title="FFN Projection Output (H_activated * W2 + b2)"
                  subtitle="Vector outputs sized back down. Shape: (seq_len, d_model)"
                />
              )}
              {ffnTab === "weights" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <MatrixHeatmap
                      matrix={data.W1}
                      rowLabels={["d0", "d1", "d2", "d3"]}
                      colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                      title="Linear Weight W1 (seed 201)"
                      subtitle="Shape: (d_model, hidden_dim)"
                    />
                    <div className="flex flex-col gap-4">
                      <MatrixHeatmap
                        matrix={[data.b1]}
                        rowLabels={["Bias b1"]}
                        colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                        title="Linear Bias b1 (seed 202)"
                        subtitle="Shape: (1, hidden_dim)"
                      />
                      <MatrixHeatmap
                        matrix={[data.b2]}
                        rowLabels={["Bias b2"]}
                        colLabels={["d0", "d1", "d2", "d3"]}
                        title="Linear Bias b2 (seed 204)"
                        subtitle="Shape: (1, d_model)"
                      />
                    </div>
                  </div>
                  <MatrixHeatmap
                    matrix={data.W2}
                    rowLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    colLabels={["d0", "d1", "d2", "d3"]}
                    title="Linear Weight W2 (seed 203)"
                    subtitle="Shape: (hidden_dim, d_model)"
                  />
                </div>
              )}
            </div>
          </div>
        );
      case 6: // Add & Norm 2
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={data.add2}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.add2[0].length }, (_, i) => `d_${i}`)}
              title="Second Residual Addition (Z_norm1 + FFN_Output)"
              subtitle="Shape: (seq_len, d_model)"
            />

            {/* LayerNorm calculations */}
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Layer Normalization Calculations (Per Row)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                      <th className="py-2 pr-4 font-sans font-semibold">Token</th>
                      <th className="py-2 pr-4 font-semibold text-center">Mean (&mu;)</th>
                      <th className="py-2 pr-4 font-semibold text-center">Variance (&sigma;<sup>2</sup>)</th>
                      <th className="py-2 font-semibold">Normalized Vector</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                    {tokens.map((token, idx) => (
                      <tr key={idx} className="text-slate-700 dark:text-slate-300">
                        <td className="py-2.5 pr-4 font-sans font-bold text-slate-800 dark:text-slate-200">{token}</td>
                        <td className="py-2.5 pr-4 text-center">{data.means2[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{data.variances2[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{data.norm2[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={data.norm2}
              rowLabels={tokens}
              colLabels={Array.from({ length: data.norm2[0].length }, (_, i) => `d_${i}`)}
              title="Final Encoder Block Layer Norm Output (norm2)"
              subtitle="Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 7: // Final Output
        return (
          <div className="flex flex-col gap-6">
            {/* Labeled contextualized vectors table */}
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-5 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4">
                Contextual Embedding Vectors Table
              </h4>
              <div className="flex flex-col gap-4">
                {tokens.map((token, idx) => (
                  <div
                    key={idx}
                    className="flex flex-col md:flex-row md:items-center justify-between gap-3 p-4 bg-slate-50/50 dark:bg-slate-900/40 rounded-xl border border-slate-200/40 dark:border-slate-800/40"
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm" />
                      <span className="font-bold text-slate-800 dark:text-slate-100 text-sm md:text-base">
                        &ldquo;{token}&rdquo;
                      </span>
                      <span className="text-[10px] text-slate-400 dark:text-slate-500 bg-slate-200/65 dark:bg-slate-800 px-2 py-0.5 rounded font-semibold font-mono">
                        pos {idx}
                      </span>
                    </div>

                    {/* Vector Display */}
                    <div className="flex items-center gap-1.5 flex-wrap font-mono">
                      {data.norm2[idx].map((val, colIdx) => (
                        <div
                          key={colIdx}
                          className="flex flex-col items-center bg-white dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800/60 rounded-lg px-3 py-1 shadow-xs min-w-[70px]"
                        >
                          <span className="text-[8px] text-slate-400 uppercase tracking-widest">
                            d_{colIdx}
                          </span>
                          <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                            {val.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <MatrixHeatmap
              matrix={data.norm2}
              rowLabels={tokens}
              colLabels={["d0", "d1", "d2", "d3"]}
              title="Final Contextualized Output Matrix (Z_norm2)"
              subtitle="The output vector block for all words. Shape: (seq_len, d_model)"
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full items-start">
      {/* Left educational panel */}
      <div className="lg:col-span-5 bg-white dark:bg-slate-950/20 border border-slate-200/60 dark:border-slate-800/60 rounded-2xl p-6 shadow-sm flex flex-col justify-between self-start min-h-[300px]">
        {renderLeftPanel()}
      </div>

      {/* Right visualization panel */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {renderRightPanel()}
      </div>
    </div>
  );
}

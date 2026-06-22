"use client";

import React, { useState } from "react";
import FormulaBlock from "./FormulaBlock";
import MatrixHeatmap from "./MatrixHeatmap";
import AttentionMap from "./AttentionMap";
import MaskGrid from "./MaskGrid";
import CrossAttentionMap from "./CrossAttentionMap";
import PredictionBars from "./PredictionBars";
import { TransformerPipelineResult, DecoderPipelineResult } from "../lib/transformer";
import { Layers, GitCommit, Database, Eye, Lock } from "lucide-react";

interface StageCardProps {
  step: number;
  encoderData: TransformerPipelineResult;
  decoderData?: DecoderPipelineResult;
  isDecoder?: boolean;
}

export default function StageCard({
  step,
  encoderData,
  decoderData,
  isDecoder = false,
}: StageCardProps) {
  // Tabs for stage sub-matrices
  const [attnTab, setAttnTab] = useState<"visualize" | "qkv" | "weights" | "mask">("visualize");
  const [crossAttnTab, setCrossAttnTab] = useState<"visualize" | "qkv" | "weights">("visualize");
  const [ffnTab, setFfnTab] = useState<"activations" | "output" | "weights">("activations");

  const data = isDecoder && decoderData ? decoderData : encoderData;
  const tokens = data.tokens;
  const vocab = data.vocab;
  const tokenIds = data.tokenIds;
  const encoderTokens = encoderData.tokens;

  // --- Encoder Left Panels ---
  const renderEncoderLeft = () => {
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
              Tokenization splits the raw input sentence into discrete word tokens by whitespace, converts them to lowercase, and assigns a unique integer index based on a dynamically built vocabulary list.
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
                <li>Weight Matrix <code className="font-mono text-indigo-500">W_embed</code>: ({vocab.length} &times; {encoderData.W_embed[0].length})</li>
                <li>Embeddings Output <code className="font-mono text-indigo-500">X_embed</code>: ({tokens.length} &times; {encoderData.embeddings[0].length})</li>
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
              Transformers process all tokens simultaneously (no recurrence). To capture the sequence order, we add a fixed positional encoding matrix <code className="font-mono text-indigo-500">PE</code> composed of sinusoidal signals. This ensures words at different positions have different vectors, encoding order without expanding vector dimensions.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Positional Encoding <code className="font-mono text-indigo-500">PE</code>: ({tokens.length} &times; {encoderData.pe[0].length})</li>
                <li>Encoded Matrix <code className="font-mono text-indigo-500">Z</code>: ({tokens.length} &times; {encoderData.Z[0].length})</li>
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
              We calculate Query-Key dot products, scale by <code className="font-mono text-indigo-500">{"1/\\sqrt{d_k}"}</code> to stabilize gradients, and apply Softmax row-wise. We multiply these weights by <code className="font-mono text-indigo-500">V</code> to compute a weighted contextual representation.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Query, Key, Value (<code className="font-mono text-indigo-500">Q, K, V</code>): ({tokens.length} &times; {encoderData.Q[0].length})</li>
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
              We add a skip connection by adding the pre-attention inputs <code className="font-mono text-indigo-500">Z</code> to the attention outputs. This alleviates vanishing gradients. We then apply Layer Normalization across the <code className="font-mono text-indigo-500">d_model</code> dimensions:
            </p>
            <FormulaBlock
              formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}"
              block={false}
              caption="Simplified LayerNorm with \gamma = 1, \beta = 0 (no learned scale/shift)"
            />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res1</code>: ({tokens.length} &times; {encoderData.add1[0].length})</li>
                <li>Normalized output <code className="font-mono text-indigo-500">Z_norm1</code>: ({tokens.length} &times; {encoderData.norm1[0].length})</li>
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
              The Feed-Forward Network (FFN) consists of two linear layers with a ReLU activation in between. It is applied to each token vector independently. The intermediate layer projects vectors to a higher-dimensional space (<code className="font-mono text-indigo-500">hidden_dim = 16</code>) to introduce non-linear mapping capacity before projecting back to <code className="font-mono text-indigo-500">d_model = 4</code>.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Hidden Matrix <code className="font-mono text-indigo-500">H_activated</code>: ({tokens.length} &times; {encoderData.W1[0].length})</li>
                <li>FFN Output: ({tokens.length} &times; {encoderData.W2[0].length})</li>
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
              caption="Second residual connection and Layer Normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add the inputs of the FFN block (<code className="font-mono text-indigo-500">Z_norm1</code>) to the outputs of the FFN block. This second residual path is then normalized using LayerNorm to yield the final output vectors of our encoder block.
            </p>
            <FormulaBlock
              formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}"
              block={false}
              caption="Simplified LayerNorm with \gamma = 1, \beta = 0 (no learned scale/shift)"
            />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res2</code>: ({tokens.length} &times; {encoderData.add2[0].length})</li>
                <li>Final Output Matrix: ({tokens.length} &times; {encoderData.norm2[0].length})</li>
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
              This is the final contextual embedding matrix produced by the Transformer Encoder Block. Every row vector represents a word in the input sequence, but unlike the static embeddings, these vectors have incorporated surrounding structural information through self-attention and non-linear mappings. These representations are ready to be passed to decoder networks.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Output Shape:</strong>
              <p className="font-mono mt-1 text-slate-600 dark:text-slate-300">
                ({tokens.length} &times; {encoderData.norm2[0].length}) [seq_len, d_model]
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // --- Decoder Left Panels ---
  const renderDecoderLeft = () => {
    if (!decoderData) return null;
    const dec = decoderData;

    switch (step) {
      case 0: // Tokenizer & Embedding
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 1
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Tokenizer & Embedding Layer
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} \text{Tokenizer}(S_{target}) &= [ID_1, ID_2, \dots, ID_T] \\ X_{embed} &= \text{Lookup}(\text{Token IDs}, W_{embed\_dec}) \end{aligned}"
              caption="Target sequence tokenization & embeddings lookup"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              The decoder takes the target sentence (capping at 6 tokens) and tokenizes it. It maps token IDs to continuous vectors using a target-specific embedding weights table <code className="font-mono text-indigo-500">W_embed_dec</code> generated with seed <code className="font-mono text-indigo-550">301</code>.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Target Tokens: {dec.tokens.length}</li>
                <li>Target Vocab: {dec.vocab.length} unique words</li>
                <li>Weight Matrix: ({dec.vocab.length} &times; {dec.W_embed[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 1: // Positional Encoding
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 2
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Positional Encoding
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ Z_{dec} &= X_{embed} + PE \end{aligned}"
              caption="Applying target positional encoding"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We apply the standard sinusoidal positional encodings to the decoder's own sequence length. Adding this PE matrix to the target embeddings yields <code className="font-mono text-indigo-500">Z_dec</code>, conveying structural sequence positions.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>PE Matrix: ({dec.tokens.length} &times; {dec.pe[0].length})</li>
                <li>Z_dec Output: ({dec.tokens.length} &times; {dec.Z[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 2: // Masked Self-Attention
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 3
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Masked Self-Attention
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} Q = Z_{dec} \cdot W_{q}, \ &K = Z_{dec} \cdot W_{k}, \ V = Z_{dec} \cdot W_{v} \\ \text{Scaled} &= \frac{Q \cdot K^T}{\sqrt{d_k}} \\ \text{Masked} &= \text{applyMask}(\text{Scaled}) \\ \text{Weights} &= \text{softmax}(\text{Masked}) \\ \text{Output} &= \text{Weights} \cdot V \end{aligned}"
              caption="Calculates masked attention and aggregates V"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We project Queries, Keys, and Values using weights with seeds <code className="font-mono text-indigo-550">311, 312, 313</code>. 
              Crucially, we apply a <strong>look-ahead mask</strong> by setting the upper-triangular elements (where key index &gt; query index) of scaled scores to <code className="font-mono text-indigo-500">-\infty</code> before applying softmax. This ensures that the representation for position $i$ only depends on positions $\le i$, preserving the auto-regressive property.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Q, K, V Projections: ({dec.tokens.length} &times; {dec.Q_self[0].length})</li>
                <li>Self-Attention Weights: ({dec.tokens.length} &times; {dec.tokens.length})</li>
              </ul>
            </div>
          </div>
        );
      case 3: // Add & Norm 1
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 4
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Add & Norm #1
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} X_{res1} &= Z_{dec} + \text{SelfAttnOutput} \\ Z_{norm1} &= \text{LayerNorm}(X_{res1}) \end{aligned}"
              caption="Residual connection and Layer Normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add the residual skip connection from pre-attention inputs <code className="font-mono text-indigo-500">Z_dec</code>, followed by Layer Normalization:
            </p>
            <FormulaBlock
              formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}"
              block={false}
              caption="Simplified LayerNorm with \gamma = 1, \beta = 0 (no learned scale/shift)"
            />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res1</code>: ({dec.tokens.length} &times; {dec.add1[0].length})</li>
                <li>Normalized Output <code className="font-mono text-indigo-500">Z_norm1</code>: ({dec.tokens.length} &times; {dec.norm1[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 4: // Encoder-Decoder Cross-Attention
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 5
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Cross-Attention
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} Q &= Z_{norm1} \cdot W_{q\_cross} \\ K &= Y_{enc\_out} \cdot W_{k\_cross}, \ V = Y_{enc\_out} \cdot W_{v\_cross} \\ \text{Scaled} &= \frac{Q \cdot K^T}{\sqrt{d_k}} \\ \text{Weights} &= \text{softmax}(\text{Scaled}) \\ \text{Output} &= \text{Weights} \cdot V \end{aligned}"
              caption="Queries decoder norm1 and references encoder output K, V"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              Encoder-Decoder cross-attention enables the decoder sequence to query the encoder sequence's final output representation. 
              The Queries (Q) come from the decoder norm1 state (projected with seed <code className="font-mono text-indigo-550">321</code>). 
              The Keys (K) and Values (V) come from the final encoder block output (projected with seeds <code className="font-mono text-indigo-550">322, 323</code>). 
              Because the encoder and decoder sequence lengths can differ, this attention weights matrix is rectangular of shape <code className="font-mono text-indigo-500">(seq_len_dec, seq_len_enc)</code>. No causal masking is applied here because the decoder is allowed to attend to the entire encoder sequence.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Decoder Q: ({dec.tokens.length} &times; {dec.Q_cross[0].length})</li>
                <li>Encoder K, V: ({encoderTokens.length} &times; {dec.K_cross[0].length})</li>
                <li>Cross-Attention Weights: ({dec.tokens.length} &times; {encoderTokens.length})</li>
              </ul>
            </div>
          </div>
        );
      case 5: // Add & Norm 2
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 6
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Add & Norm #2
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} X_{res2} &= Z_{norm1} + \text{CrossAttnOutput} \\ Z_{norm2} &= \text{LayerNorm}(X_{res2}) \end{aligned}"
              caption="Residual connection and Layer Normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add the cross-attention output back to its pre-attention inputs <code className="font-mono text-indigo-500">Z_norm1</code>, and apply Layer Normalization:
            </p>
            <FormulaBlock
              formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}"
              block={false}
              caption="Simplified LayerNorm with \gamma = 1, \beta = 0 (no learned scale/shift)"
            />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res2</code>: ({dec.tokens.length} &times; {dec.add2[0].length})</li>
                <li>Normalized Output <code className="font-mono text-indigo-500">Z_norm2</code>: ({dec.tokens.length} &times; {dec.norm2[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 6: // Feed-Forward Network
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 7
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Feed-Forward Network
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} H_{raw} &= X \cdot W_{1} + b_{1} \\ H_{activated} &= \max(0, H_{raw}) \\ \text{FFN\_Output} &= H_{activated} \cdot W_{2} + b_{2} \end{aligned}"
              caption="Decoder two-layer FFN projection with ReLU activation"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We apply FFN projection to each decoder token vector independently. Weight matrices use seeds <code className="font-mono text-indigo-550">331</code> and <code className="font-mono text-indigo-550">333</code>. The FFN bias vectors <code className="font-mono text-indigo-500">b1</code> (seed <code className="font-mono text-indigo-550">332</code>) and <code className="font-mono text-indigo-500">b2</code> (seed <code className="font-mono text-indigo-550">334</code>) are scaled to a small range <code className="font-mono text-indigo-500">[-0.2, 0.2]</code> so they do not dominate FFN activations.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Hidden Matrix: ({dec.tokens.length} &times; {dec.W1[0].length})</li>
                <li>FFN Output: ({dec.tokens.length} &times; {dec.W2[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 7: // Add & Norm 3
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 8
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Add & Norm #3
              </h3>
            </div>
            <FormulaBlock
              formula="\begin{aligned} X_{res3} &= Z_{norm2} + \text{FFN\_Output} \\ Z_{norm3} &= \text{LayerNorm}(X_{res3}) \end{aligned}"
              caption="Third residual connection and Layer Normalization"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We add the FFN output back to its input vectors <code className="font-mono text-indigo-500">Z_norm2</code>, and apply Layer Normalization to produce the final decoder block outputs:
            </p>
            <FormulaBlock
              formula="\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}"
              block={false}
              caption="Simplified LayerNorm with \gamma = 1, \beta = 0 (no learned scale/shift)"
            />
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Residual Sum <code className="font-mono text-indigo-500">X_res3</code>: ({dec.tokens.length} &times; {dec.add3[0].length})</li>
                <li>Final Decoder Output <code className="font-mono text-indigo-500">Z_norm3</code>: ({dec.tokens.length} &times; {dec.norm3[0].length})</li>
              </ul>
            </div>
          </div>
        );
      case 8: // Output Projection
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 9
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Output Projection (Logits)
              </h3>
            </div>
            <FormulaBlock
              formula="\text{Logits} = Z_{norm3} \cdot W_{proj}"
              caption="Projecting output embeddings back to vocabulary logits"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              To convert continuous vectors back into token predictions, we project the final decoder vectors to the target vocabulary space using a projection weight matrix <code className="font-mono text-indigo-500">W_proj</code> of shape <code className="font-mono text-indigo-500">(d_model, vocab_size)</code> with seed <code className="font-mono text-indigo-550">341</code>.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Dimensions:</strong>
              <ul className="list-disc pl-4 space-y-1">
                <li>Decoder Output shape: ({dec.tokens.length} &times; {dec.norm3[0].length})</li>
                <li>Projection Weights `W_proj`: ({dec.norm3[0].length} &times; {dec.vocab.length})</li>
                <li>Logits Output shape: ({dec.tokens.length} &times; {dec.vocab.length})</li>
              </ul>
            </div>
          </div>
        );
      case 9: // Softmax & Prediction
        return (
          <div className="flex flex-col gap-4">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Stage 10
              </span>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mt-1">
                Softmax & Prediction
              </h3>
            </div>
            <FormulaBlock
              formula="\text{Probabilities} = \text{softmax}(\text{Logits})"
              caption="Calculating next-token probabilities"
            />
            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              We apply a row-wise Softmax over the raw logits to calculate a probability distribution for each position. In generative models, the prediction at the <strong>last token position</strong> (pos {dec.tokens.length - 1}) represents the model's prediction for the next word in the sequence.
            </p>
            <div className="bg-slate-100/60 dark:bg-slate-900/50 rounded-xl p-3 border border-slate-200/40 dark:border-slate-800/40 text-xs text-slate-500 dark:text-slate-400">
              <strong className="text-slate-700 dark:text-slate-300 block mb-1">Final Token argmax prediction:</strong>
              <p className="mt-1 font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                Selects the vocabulary word with highest probability.
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // --- Encoder Right Panels ---
  const renderEncoderRight = () => {
    switch (step) {
      case 0: // Tokenizer
        return (
          <div className="flex flex-col gap-5">
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
              matrix={encoderData.W_embed}
              rowLabels={vocab.map((v, idx) => `ID ${idx} [${v}]`)}
              colLabels={Array.from({ length: encoderData.W_embed[0].length }, (_, i) => `d_${i}`)}
              title="Embedding Weight Matrix (W_embed)"
              subtitle="Weight values generated with seed 42. Shape: (vocab_size, d_model)"
            />
            <MatrixHeatmap
              matrix={encoderData.embeddings}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.embeddings[0].length }, (_, i) => `d_${i}`)}
              title="Looked Up Token Embeddings (X_embed)"
              subtitle="Values grabbed from matching rows of W_embed. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 2: // Positional Encoding
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={encoderData.pe}
              rowLabels={tokens.map((t, i) => `${t} (pos ${i})`)}
              colLabels={Array.from({ length: encoderData.pe[0].length }, (_, i) => `d_${i}`)}
              title="Positional Encoding Matrix (PE)"
              subtitle="Calculated sine & cosine constants. Shape: (seq_len, d_model)"
            />
            <MatrixHeatmap
              matrix={encoderData.Z}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.Z[0].length }, (_, i) => `d_${i}`)}
              title="Encoded Embeddings Input (Z = X_embed + PE)"
              subtitle="Combined vectors containing both semantic and index context. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 3: // Self-Attention
        return (
          <div className="flex flex-col gap-4">
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
                <AttentionMap tokens={tokens} attentionWeights={encoderData.attentionWeights} />
              )}
              {attnTab === "qkv" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <MatrixHeatmap
                      matrix={encoderData.Q}
                      rowLabels={tokens}
                      colLabels={["q0", "q1", "q2", "q3"]}
                      title="Queries (Q = Z * Wq)"
                    />
                    <MatrixHeatmap
                      matrix={encoderData.K}
                      rowLabels={tokens}
                      colLabels={["k0", "k1", "k2", "k3"]}
                      title="Keys (K = Z * Wk)"
                    />
                    <MatrixHeatmap
                      matrix={encoderData.V}
                      rowLabels={tokens}
                      colLabels={["v0", "v1", "v2", "v3"]}
                      title="Values (V = Z * Wv)"
                    />
                  </div>
                  <MatrixHeatmap
                    matrix={encoderData.attentionOutput}
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
                    matrix={encoderData.Wq}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["q0", "q1", "q2", "q3"]}
                    title="Wq Matrix (seed 101)"
                  />
                  <MatrixHeatmap
                    matrix={encoderData.Wk}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["k0", "k1", "k2", "k3"]}
                    title="Wk Matrix (seed 102)"
                  />
                  <MatrixHeatmap
                    matrix={encoderData.Wv}
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
              matrix={encoderData.add1}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.add1[0].length }, (_, i) => `d_${i}`)}
              title="Residual Addition (Z + AttentionOutput)"
              subtitle="Shape: (seq_len, d_model)"
            />

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
                        <td className="py-2.5 pr-4 text-center">{encoderData.means1[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{encoderData.variances1[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{encoderData.norm1[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={encoderData.norm1}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.norm1[0].length }, (_, i) => `d_${i}`)}
              title="Layer Normalization Output (Z_norm1)"
              subtitle="Output scaled to mean=0, variance=1. Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 5: // Feed Forward
        return (
          <div className="flex flex-col gap-4">
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
                    matrix={encoderData.H_raw}
                    rowLabels={tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Pre-Activation Hidden Matrix (H_raw = Z_norm1 * W1 + b1)"
                    subtitle="Calculated dot-products. Shape: (seq_len, hidden_dim)"
                  />
                  <MatrixHeatmap
                    matrix={encoderData.H_activated}
                    rowLabels={tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Intermediate Hidden Activation Matrix (H_activated = ReLU(H_raw))"
                    subtitle="Values below 0 are zeroed out (ReLU). Shape: (seq_len, hidden_dim)"
                  />
                </div>
              )}
              {ffnTab === "output" && (
                <MatrixHeatmap
                  matrix={encoderData.ffnOutput}
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
                      matrix={encoderData.W1}
                      rowLabels={["d0", "d1", "d2", "d3"]}
                      colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                      title="Linear Weight W1 (seed 201)"
                      subtitle="Shape: (d_model, hidden_dim)"
                    />
                    <div className="flex flex-col gap-4">
                      <MatrixHeatmap
                        matrix={[encoderData.b1]}
                        rowLabels={["Bias b1"]}
                        colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                        title="Bias b1 (seed 202)"
                        subtitle="Shape: (1, hidden_dim)"
                      />
                      <MatrixHeatmap
                        matrix={[encoderData.b2]}
                        rowLabels={["Bias b2"]}
                        colLabels={Array.from({ length: 16 }, (_, i) => `d_${i}`)}
                        title="Bias b2 (seed 204)"
                        subtitle="Shape: (1, d_model)"
                      />
                    </div>
                  </div>
                  <MatrixHeatmap
                    matrix={encoderData.W2}
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
              matrix={encoderData.add2}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.add2[0].length }, (_, i) => `d_${i}`)}
              title="Second Residual Addition (Z_norm1 + FFN_Output)"
              subtitle="Shape: (seq_len, d_model)"
            />

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
                        <td className="py-2.5 pr-4 text-center">{encoderData.means2[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{encoderData.variances2[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{encoderData.norm2[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={encoderData.norm2}
              rowLabels={tokens}
              colLabels={Array.from({ length: encoderData.norm2[0].length }, (_, i) => `d_${i}`)}
              title="Final Encoder Block Layer Norm Output (norm2)"
              subtitle="Shape: (seq_len, d_model)"
            />
          </div>
        );
      case 7: // Final Output
        return (
          <div className="flex flex-col gap-6">
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

                    <div className="flex items-center gap-1.5 flex-wrap font-mono">
                      {encoderData.norm2[idx].map((val, colIdx) => (
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
              matrix={encoderData.norm2}
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

  // --- Decoder Right Panels ---
  const renderDecoderRight = () => {
    if (!decoderData) return null;
    const dec = decoderData;

    switch (step) {
      case 0: // Tokenizer & Embedding
        return (
          <div className="flex flex-col gap-5">
            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Decoder Vocabulary List
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {dec.vocab.map((v, i) => (
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

            <div className="bg-white dark:bg-slate-950/40 border border-slate-200/60 dark:border-slate-800/60 rounded-xl p-4 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
                Decoder Token Mapping (Target sentence &rarr; IDs)
              </h4>
              <div className="flex flex-wrap items-center gap-2">
                {dec.tokens.map((token, idx) => (
                  <div key={idx} className="flex items-center">
                    <div className="flex flex-col items-center bg-slate-50 dark:bg-slate-900 px-4 py-2 border border-slate-200/60 dark:border-slate-800/60 rounded-xl shadow-xs">
                      <span className="text-[10px] text-slate-400 dark:text-slate-500">
                        pos {idx}
                      </span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {token}
                      </span>
                      <span className="font-bold text-indigo-500 font-mono text-xs mt-1 bg-indigo-50 dark:bg-indigo-950/50 px-2 py-0.5 rounded">
                        {dec.tokenIds[idx]}
                      </span>
                    </div>
                    {idx < dec.tokens.length - 1 && (
                      <span className="mx-1.5 text-slate-300 dark:text-slate-700 font-bold">&rarr;</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <MatrixHeatmap
              matrix={dec.W_embed}
              rowLabels={dec.vocab.map((v, idx) => `ID ${idx} [${v}]`)}
              colLabels={Array.from({ length: dec.W_embed[0].length }, (_, i) => `d_${i}`)}
              title="Decoder Embedding Weights (W_embed_dec)"
              subtitle="Weight values generated with seed 301. Shape: (vocab_size_dec, d_model)"
            />
            <MatrixHeatmap
              matrix={dec.embeddings}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.embeddings[0].length }, (_, i) => `d_${i}`)}
              title="Looked Up Token Embeddings (X_embed_dec)"
              subtitle="Looked up from W_embed_dec. Shape: (seq_len_dec, d_model)"
            />
          </div>
        );
      case 1: // PE
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={dec.pe}
              rowLabels={dec.tokens.map((t, i) => `${t} (pos ${i})`)}
              colLabels={Array.from({ length: dec.pe[0].length }, (_, i) => `d_${i}`)}
              title="Decoder Positional Encoding Matrix (PE)"
              subtitle="Calculated sinusoidal PE. Shape: (seq_len_dec, d_model)"
            />
            <MatrixHeatmap
              matrix={dec.Z}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.Z[0].length }, (_, i) => `d_${i}`)}
              title="Decoder Encoded Embeddings (Z_dec = X_embed + PE)"
              subtitle="Combined vectors. Shape: (seq_len_dec, d_model)"
            />
          </div>
        );
      case 2: // Masked Self-Attention
        return (
          <div className="flex flex-col gap-4">
            <div className="flex border-b border-slate-200 dark:border-slate-800">
              <button
                onClick={() => setAttnTab("mask")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  attnTab === "mask"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Lock className="w-3.5 h-3.5" />
                <span>Look-Ahead Mask</span>
              </button>
              <button
                onClick={() => setAttnTab("visualize")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  attnTab === "visualize"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-650 dark:hover:text-slate-300"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Masked Attention Maps</span>
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
              {attnTab === "mask" && <MaskGrid tokens={dec.tokens} />}
              {attnTab === "visualize" && (
                <AttentionMap tokens={dec.tokens} attentionWeights={dec.attentionWeights_self} />
              )}
              {attnTab === "qkv" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <MatrixHeatmap
                      matrix={dec.Q_self}
                      rowLabels={dec.tokens}
                      colLabels={["q0", "q1", "q2", "q3"]}
                      title="Queries (Q = Z * Wq_self)"
                    />
                    <MatrixHeatmap
                      matrix={dec.K_self}
                      rowLabels={dec.tokens}
                      colLabels={["k0", "k1", "k2", "k3"]}
                      title="Keys (K = Z * Wk_self)"
                    />
                    <MatrixHeatmap
                      matrix={dec.V_self}
                      rowLabels={dec.tokens}
                      colLabels={["v0", "v1", "v2", "v3"]}
                      title="Values (V = Z * Wv_self)"
                    />
                  </div>
                  <MatrixHeatmap
                    matrix={dec.attentionOutput_self}
                    rowLabels={dec.tokens}
                    colLabels={["d0", "d1", "d2", "d3"]}
                    title="Self-Attention Output (Weights * V)"
                    subtitle="Weighted sum of V. Shape: (seq_len_dec, d_model)"
                  />
                </div>
              )}
              {attnTab === "weights" && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MatrixHeatmap
                    matrix={dec.Wq_self}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["q0", "q1", "q2", "q3"]}
                    title="Wq_self Weights (seed 311)"
                  />
                  <MatrixHeatmap
                    matrix={dec.Wk_self}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["k0", "k1", "k2", "k3"]}
                    title="Wk_self Weights (seed 312)"
                  />
                  <MatrixHeatmap
                    matrix={dec.Wv_self}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["v0", "v1", "v2", "v3"]}
                    title="Wv_self Weights (seed 313)"
                  />
                </div>
              )}
            </div>
          </div>
        );
      case 3: // Add & Norm 1
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={dec.add1}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.add1[0].length }, (_, i) => `d_${i}`)}
              title="Residual Addition (Z_dec + SelfAttnOutput)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />

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
                    {dec.tokens.map((token, idx) => (
                      <tr key={idx} className="text-slate-700 dark:text-slate-300">
                        <td className="py-2.5 pr-4 font-sans font-bold text-slate-800 dark:text-slate-200">{token}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.means1[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.variances1[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{dec.norm1[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={dec.norm1}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.norm1[0].length }, (_, i) => `d_${i}`)}
              title="Layer Normalization Output (Z_norm1)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />
          </div>
        );
      case 4: // Encoder-Decoder Cross-Attention
        return (
          <div className="flex flex-col gap-4">
            <div className="flex border-b border-slate-200 dark:border-slate-800">
              <button
                onClick={() => setCrossAttnTab("visualize")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  crossAttnTab === "visualize"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Cross-Attention Map</span>
              </button>
              <button
                onClick={() => setCrossAttnTab("qkv")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  crossAttnTab === "qkv"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-650 dark:hover:text-slate-300"
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>Q, K, V Projections</span>
              </button>
              <button
                onClick={() => setCrossAttnTab("weights")}
                className={`pb-2.5 px-4 text-xs font-semibold flex items-center gap-1.5 border-b-2 cursor-pointer transition-colors duration-150 ${
                  crossAttnTab === "weights"
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Wq, Wk, Wv Weights</span>
              </button>
            </div>

            <div className="mt-2">
              {crossAttnTab === "visualize" && (
                <CrossAttentionMap
                  attentionWeights={dec.attentionWeights_cross}
                  decoderTokens={dec.tokens}
                  encoderTokens={encoderTokens}
                />
              )}
              {crossAttnTab === "qkv" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <MatrixHeatmap
                      matrix={dec.Q_cross}
                      rowLabels={dec.tokens}
                      colLabels={["q0", "q1", "q2", "q3"]}
                      title="Queries (Q = Z_norm1 * Wq_cross)"
                      subtitle="From Decoder sequence"
                    />
                    <MatrixHeatmap
                      matrix={dec.K_cross}
                      rowLabels={encoderTokens}
                      colLabels={["k0", "k1", "k2", "k3"]}
                      title="Keys (K = encoderOutput * Wk_cross)"
                      subtitle="From Encoder sequence output"
                    />
                    <MatrixHeatmap
                      matrix={dec.V_cross}
                      rowLabels={encoderTokens}
                      colLabels={["v0", "v1", "v2", "v3"]}
                      title="Values (V = encoderOutput * Wv_cross)"
                      subtitle="From Encoder sequence output"
                    />
                  </div>
                  <MatrixHeatmap
                    matrix={dec.attentionOutput_cross}
                    rowLabels={dec.tokens}
                    colLabels={["d0", "d1", "d2", "d3"]}
                    title="Cross-Attention Output (Weights * V)"
                    subtitle="Weighted sum of Encoder V. Shape: (seq_len_dec, d_model)"
                  />
                </div>
              )}
              {crossAttnTab === "weights" && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MatrixHeatmap
                    matrix={dec.Wq_cross}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["q0", "q1", "q2", "q3"]}
                    title="Wq_cross Weights (seed 321)"
                  />
                  <MatrixHeatmap
                    matrix={dec.Wk_cross}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["k0", "k1", "k2", "k3"]}
                    title="Wk_cross Weights (seed 322)"
                  />
                  <MatrixHeatmap
                    matrix={dec.Wv_cross}
                    rowLabels={["d0", "d1", "d2", "d3"]}
                    colLabels={["v0", "v1", "v2", "v3"]}
                    title="Wv_cross Weights (seed 323)"
                  />
                </div>
              )}
            </div>
          </div>
        );
      case 5: // Add & Norm 2
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={dec.add2}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.add2[0].length }, (_, i) => `d_${i}`)}
              title="Residual Addition (Z_norm1 + CrossAttnOutput)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />

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
                    {dec.tokens.map((token, idx) => (
                      <tr key={idx} className="text-slate-700 dark:text-slate-300">
                        <td className="py-2.5 pr-4 font-sans font-bold text-slate-800 dark:text-slate-200">{token}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.means2[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.variances2[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{dec.norm2[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={dec.norm2}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.norm2[0].length }, (_, i) => `d_${i}`)}
              title="Layer Normalization Output (Z_norm2)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />
          </div>
        );
      case 6: // Feed-Forward Network
        return (
          <div className="flex flex-col gap-4">
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
                    : "border-transparent text-slate-400 hover:text-slate-650 dark:hover:text-slate-350"
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
                    matrix={dec.H_raw}
                    rowLabels={dec.tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Decoder Pre-Activation (H_raw = Z_norm2 * W1 + b1)"
                    subtitle="Shape: (seq_len_dec, hidden_dim)"
                  />
                  <MatrixHeatmap
                    matrix={dec.H_activated}
                    rowLabels={dec.tokens}
                    colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    title="Decoder Hidden Activation (H_activated = ReLU(H_raw))"
                    subtitle="Shape: (seq_len_dec, hidden_dim)"
                  />
                </div>
              )}
              {ffnTab === "output" && (
                <MatrixHeatmap
                  matrix={dec.ffnOutput}
                  rowLabels={dec.tokens}
                  colLabels={Array.from({ length: 4 }, (_, i) => `d_${i}`)}
                  title="Decoder FFN Output (H_activated * W2 + b2)"
                  subtitle="Shape: (seq_len_dec, d_model)"
                />
              )}
              {ffnTab === "weights" && (
                <div className="flex flex-col gap-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <MatrixHeatmap
                      matrix={dec.W1}
                      rowLabels={["d0", "d1", "d2", "d3"]}
                      colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                      title="Linear Weight W1 (seed 331)"
                      subtitle="Shape: (d_model, hidden_dim)"
                    />
                    <div className="flex flex-col gap-4">
                      <MatrixHeatmap
                        matrix={[dec.b1]}
                        rowLabels={["Bias b1"]}
                        colLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                        title="Bias b1 (seed 332)"
                        subtitle="Shape: (1, hidden_dim)"
                      />
                      <MatrixHeatmap
                        matrix={[dec.b2]}
                        rowLabels={["Bias b2"]}
                        colLabels={Array.from({ length: 16 }, (_, i) => `d_${i}`)}
                        title="Bias b2 (seed 334)"
                        subtitle="Shape: (1, d_model)"
                      />
                    </div>
                  </div>
                  <MatrixHeatmap
                    matrix={dec.W2}
                    rowLabels={Array.from({ length: 16 }, (_, i) => `h_${i}`)}
                    colLabels={["d0", "d1", "d2", "d3"]}
                    title="Linear Weight W2 (seed 333)"
                    subtitle="Shape: (hidden_dim, d_model)"
                  />
                </div>
              )}
            </div>
          </div>
        );
      case 7: // Add & Norm 3
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={dec.add3}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.add3[0].length }, (_, i) => `d_${i}`)}
              title="Third Residual Addition (Z_norm2 + FFN_Output)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />

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
                    {dec.tokens.map((token, idx) => (
                      <tr key={idx} className="text-slate-700 dark:text-slate-300">
                        <td className="py-2.5 pr-4 font-sans font-bold text-slate-800 dark:text-slate-200">{token}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.means3[idx].toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-center">{dec.variances3[idx].toFixed(4)}</td>
                        <td className="py-2.5 font-semibold text-indigo-600 dark:text-indigo-400">
                          [{dec.norm3[idx].map((val) => val.toFixed(2)).join(", ")}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MatrixHeatmap
              matrix={dec.norm3}
              rowLabels={dec.tokens}
              colLabels={Array.from({ length: dec.norm3[0].length }, (_, i) => `d_${i}`)}
              title="Decoder Layer Norm Output (norm3)"
              subtitle="Shape: (seq_len_dec, d_model)"
            />
          </div>
        );
      case 8: // Output Projection
        return (
          <div className="flex flex-col gap-6">
            <MatrixHeatmap
              matrix={dec.W_proj}
              rowLabels={["d0", "d1", "d2", "d3"]}
              colLabels={dec.vocab.map((v, idx) => `ID ${idx} [${v}]`)}
              title="Output Projection Matrix (W_proj)"
              subtitle="Seeded random projection weights (seed 341). Shape: (d_model, vocab_size_dec)"
            />
            <MatrixHeatmap
              matrix={dec.logits}
              rowLabels={dec.tokens}
              colLabels={dec.vocab}
              title="Calculated Raw Logits"
              subtitle="decoderOutput * W_proj. Shape: (seq_len_dec, vocab_size_dec)"
            />
          </div>
        );
      case 9: // Softmax & Prediction
        return (
          <div className="flex flex-col gap-6">
            <PredictionBars probabilities={dec.probabilities} vocab={dec.vocab} />
            <MatrixHeatmap
              matrix={dec.probabilities}
              rowLabels={dec.tokens}
              colLabels={dec.vocab}
              title="Softmax Probability Distribution Table"
              subtitle="Row-wise softmax applied to logits. Shape: (seq_len_dec, vocab_size_dec)"
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
        {isDecoder ? renderDecoderLeft() : renderEncoderLeft()}
      </div>

      {/* Right visualization panel */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {isDecoder ? renderDecoderRight() : renderEncoderRight()}
      </div>
    </div>
  );
}

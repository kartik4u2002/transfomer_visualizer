import {
  randomMatrix,
  matmul,
  transpose,
  softmax,
  layerNorm,
  add,
  relu,
  scale,
  applyMask,
} from "./matrixUtils";

export interface TransformerPipelineResult {
  tokens: string[];
  vocab: string[];
  tokenIds: number[];

  // Stage 2: Embedding
  W_embed: number[][];
  embeddings: number[][];

  // Stage 3: Positional Encoding
  pe: number[][];
  Z: number[][];

  // Stage 4: Self-Attention
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Q: number[][];
  K: number[][];
  V: number[][];
  rawScores: number[][];
  scaledScores: number[][];
  attentionWeights: number[][];
  attentionOutput: number[][];

  // Stage 5: Add & Norm 1
  add1: number[][];
  means1: number[];
  variances1: number[];
  norm1: number[][];

  // Stage 6: Feed Forward
  W1: number[][];
  b1: number[];
  H_raw: number[][];
  H_activated: number[][];
  W2: number[][];
  b2: number[];
  ffnOutput: number[][];

  // Stage 7: Add & Norm 2
  add2: number[][];
  means2: number[];
  variances2: number[];
  norm2: number[][];
}

/**
 * Runs a single Transformer Encoder Block on the input sentence.
 * All operations are deterministic, using seeded random weights.
 */
export function runTransformerPipeline(
  sentence: string,
  d_model = 4,
  hidden_dim = 16
): TransformerPipelineResult {
  // --- Stage 1: Tokenizer ---
  // Lowercase, trim, and split by whitespace
  const tokens = sentence
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0);

  // Build vocab dynamically from the input sentence
  const vocab: string[] = [];
  tokens.forEach((token) => {
    if (!vocab.includes(token)) {
      vocab.push(token);
    }
  });

  const vocabSize = vocab.length || 1; // Prevent 0 vocab size
  const tokenIds = tokens.map((token) => vocab.indexOf(token));
  const seqLen = tokens.length || 1;

  // --- Stage 2: Embedding Layer ---
  // Seeded random embedding matrix of shape (vocab_size, d_model)
  // We use seed 42
  const W_embed = randomMatrix(vocabSize, d_model, 42, -1.0, 1.0);

  // Look up embedding for each token
  const embeddings: number[][] = [];
  for (let t = 0; t < seqLen; t++) {
    const id = tokenIds[t] !== -1 ? tokenIds[t] : 0;
    // Copy the row
    embeddings.push([...W_embed[id]]);
  }

  // --- Stage 3: Positional Encoding ---
  const pe: number[][] = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row: number[] = [];
    for (let i = 0; i < d_model; i++) {
      const exponent = (2 * Math.floor(i / 2)) / d_model;
      const divisor = Math.pow(10000, exponent);
      if (i % 2 === 0) {
        row.push(Math.sin(pos / divisor));
      } else {
        row.push(Math.cos(pos / divisor));
      }
    }
    pe.push(row);
  }

  // Z = Embeddings + Positional Encoding
  const Z = add(embeddings, pe);

  // --- Stage 4: Self-Attention ---
  // Seeded weights for projections
  const Wq = randomMatrix(d_model, d_model, 101, -0.8, 0.8);
  const Wk = randomMatrix(d_model, d_model, 102, -0.8, 0.8);
  const Wv = randomMatrix(d_model, d_model, 103, -0.8, 0.8);

  // Projections
  const Q = matmul(Z, Wq);
  const K = matmul(Z, Wk);
  const V = matmul(Z, Wv);

  // Scores = Q * K^T
  const K_T = transpose(K);
  const rawScores = matmul(Q, K_T);

  // Scaled = Scores / sqrt(d_k)
  const d_k = d_model; // d_k equals d_model for single head
  const scaleFactor = 1 / Math.sqrt(d_k);
  const scaledScores = scale(rawScores, scaleFactor);

  // Weights = softmax(Scaled)
  const attentionWeights = softmax(scaledScores);

  // AttentionOutput = Weights * V
  const attentionOutput = matmul(attentionWeights, V);

  // --- Stage 5: Add & Norm #1 ---
  const add1 = add(Z, attentionOutput);
  const { output: norm1, means: means1, variances: variances1 } = layerNorm(add1);

  // --- Stage 6: Feed-Forward Network ---
  // Linear (d_model -> hidden_dim) -> ReLU -> Linear (hidden_dim -> d_model)
  const W1 = randomMatrix(d_model, hidden_dim, 201, -0.8, 0.8);
  // Bias b1 as an array of size hidden_dim
  const randB1 = randomMatrix(1, hidden_dim, 202, -0.2, 0.2)[0];
  const b1 = randB1;

  const W2 = randomMatrix(hidden_dim, d_model, 203, -0.8, 0.8);
  // Bias b2 as an array of size d_model
  const randB2 = randomMatrix(1, d_model, 204, -0.2, 0.2)[0];
  const b2 = randB2;

  // H_raw = Z_norm1 * W1 + b1
  const proj1 = matmul(norm1, W1);
  const H_raw = proj1.map((row) => row.map((val, c) => val + b1[c]));

  // H_activated = ReLU(H_raw)
  const H_activated = relu(H_raw);

  // FFN_output = H_activated * W2 + b2
  const proj2 = matmul(H_activated, W2);
  const ffnOutput = proj2.map((row) => row.map((val, c) => val + b2[c]));

  // --- Stage 7: Add & Norm #2 ---
  const add2 = add(norm1, ffnOutput);
  const { output: norm2, means: means2, variances: variances2 } = layerNorm(add2);

  return {
    tokens,
    vocab,
    tokenIds,
    W_embed,
    embeddings,
    pe,
    Z,
    Wq,
    Wk,
    Wv,
    Q,
    K,
    V,
    rawScores,
    scaledScores,
    attentionWeights,
    attentionOutput,
    add1,
    means1,
    variances1,
    norm1,
    W1,
    b1,
    H_raw,
    H_activated,
    W2,
    b2,
    ffnOutput,
    add2,
    means2,
    variances2,
    norm2,
  };
}

export interface DecoderPipelineResult {
  tokens: string[];
  vocab: string[];
  tokenIds: number[];

  // Stage 1: Embedding
  W_embed: number[][];
  embeddings: number[][];

  // Stage 2: PE
  pe: number[][];
  Z: number[][];

  // Stage 3: Masked Self-Attention
  Wq_self: number[][];
  Wk_self: number[][];
  Wv_self: number[][];
  Q_self: number[][];
  K_self: number[][];
  V_self: number[][];
  rawScores_self: number[][];
  scaledScores_self: number[][];
  maskedScores_self: number[][];
  attentionWeights_self: number[][];
  attentionOutput_self: number[][];

  // Stage 4: Add & Norm 1
  add1: number[][];
  means1: number[];
  variances1: number[];
  norm1: number[][];

  // Stage 5: Cross-Attention
  Wq_cross: number[][];
  Wk_cross: number[][];
  Wv_cross: number[][];
  Q_cross: number[][];
  K_cross: number[][];
  V_cross: number[][];
  rawScores_cross: number[][];
  scaledScores_cross: number[][];
  attentionWeights_cross: number[][];
  attentionOutput_cross: number[][];

  // Stage 6: Add & Norm 2
  add2: number[][];
  means2: number[];
  variances2: number[];
  norm2: number[][];

  // Stage 7: FFN
  W1: number[][];
  b1: number[];
  H_raw: number[][];
  H_activated: number[][];
  W2: number[][];
  b2: number[];
  ffnOutput: number[][];

  // Stage 8: Add & Norm 3
  add3: number[][];
  means3: number[];
  variances3: number[];
  norm3: number[][];

  // Stage 9: Output Projection
  W_proj: number[][];
  logits: number[][];

  // Stage 10: Softmax & Prediction
  probabilities: number[][];
}

/**
 * Runs a single Transformer Decoder Block.
 * All operations are deterministic, using seeded random weights.
 */
export function runDecoderPipeline(
  targetSentence: string,
  encoderOutput: number[][],
  d_model = 4,
  hidden_dim = 16
): DecoderPipelineResult {
  // --- Stage 1: Tokenizer & Embedding ---
  // Tokenize and cap at 6 tokens
  const tokens = targetSentence
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0)
    .slice(0, 6);

  // Build target vocab dynamically
  const vocab: string[] = [];
  tokens.forEach((token) => {
    if (!vocab.includes(token)) {
      vocab.push(token);
    }
  });

  const vocabSize = vocab.length || 1;
  const tokenIds = tokens.map((token) => vocab.indexOf(token));
  const seqLen = tokens.length || 1;

  // Seeded random embedding matrix with seed 301
  const W_embed = randomMatrix(vocabSize, d_model, 301, -1.0, 1.0);

  // Look up embedding for each token
  const embeddings: number[][] = [];
  for (let t = 0; t < seqLen; t++) {
    const id = tokenIds[t] !== -1 ? tokenIds[t] : 0;
    embeddings.push([...W_embed[id]]);
  }

  // --- Stage 2: Positional Encoding ---
  const pe: number[][] = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row: number[] = [];
    for (let i = 0; i < d_model; i++) {
      const exponent = (2 * Math.floor(i / 2)) / d_model;
      const divisor = Math.pow(10000, exponent);
      if (i % 2 === 0) {
        row.push(Math.sin(pos / divisor));
      } else {
        row.push(Math.cos(pos / divisor));
      }
    }
    pe.push(row);
  }

  // Z_dec = Embeddings + PE
  const Z = add(embeddings, pe);

  // --- Stage 3: Masked Self-Attention ---
  const Wq_self = randomMatrix(d_model, d_model, 311, -0.8, 0.8);
  const Wk_self = randomMatrix(d_model, d_model, 312, -0.8, 0.8);
  const Wv_self = randomMatrix(d_model, d_model, 313, -0.8, 0.8);

  const Q_self = matmul(Z, Wq_self);
  const K_self = matmul(Z, Wk_self);
  const V_self = matmul(Z, Wv_self);

  // Scores = Q * K_self^T
  const rawScores_self = matmul(Q_self, transpose(K_self));

  // Scale = Scores / sqrt(d_model)
  const scaleFactor = 1 / Math.sqrt(d_model);
  const scaledScores_self = scale(rawScores_self, scaleFactor);

  // Apply look-ahead mask (upper triangular set to -Infinity)
  const maskedScores_self = applyMask(scaledScores_self);

  // Softmax over masked scores
  const attentionWeights_self = softmax(maskedScores_self);

  // Output = Weights * V
  const attentionOutput_self = matmul(attentionWeights_self, V_self);

  // --- Stage 4: Add & Norm 1 ---
  const add1 = add(Z, attentionOutput_self);
  const { output: norm1, means: means1, variances: variances1 } = layerNorm(add1);

  // --- Stage 5: Encoder-Decoder Cross-Attention ---
  const Wq_cross = randomMatrix(d_model, d_model, 321, -0.8, 0.8);
  const Wk_cross = randomMatrix(d_model, d_model, 322, -0.8, 0.8);
  const Wv_cross = randomMatrix(d_model, d_model, 323, -0.8, 0.8);

  // Q is from decoder norm1; K and V are from encoderOutput
  const Q_cross = matmul(norm1, Wq_cross);
  const K_cross = matmul(encoderOutput, Wk_cross);
  const V_cross = matmul(encoderOutput, Wv_cross);

  // Scores = Q_cross * K_cross^T
  const rawScores_cross = matmul(Q_cross, transpose(K_cross));
  const scaledScores_cross = scale(rawScores_cross, scaleFactor);

  // Softmax (no mask for cross-attention)
  const attentionWeights_cross = softmax(scaledScores_cross);
  const attentionOutput_cross = matmul(attentionWeights_cross, V_cross);

  // --- Stage 6: Add & Norm 2 ---
  const add2 = add(norm1, attentionOutput_cross);
  const { output: norm2, means: means2, variances: variances2 } = layerNorm(add2);

  // --- Stage 7: Feed-Forward Network ---
  const W1 = randomMatrix(d_model, hidden_dim, 331, -0.8, 0.8);
  const b1 = randomMatrix(1, hidden_dim, 332, -0.2, 0.2)[0];
  const W2 = randomMatrix(hidden_dim, d_model, 333, -0.8, 0.8);
  const b2 = randomMatrix(1, d_model, 334, -0.2, 0.2)[0];

  // H_raw = norm2 * W1 + b1
  const proj1 = matmul(norm2, W1);
  const H_raw = proj1.map((row) => row.map((val, c) => val + b1[c]));
  const H_activated = relu(H_raw);

  // ffnOutput = H_activated * W2 + b2
  const proj2 = matmul(H_activated, W2);
  const ffnOutput = proj2.map((row) => row.map((val, c) => val + b2[c]));

  // --- Stage 8: Add & Norm 3 ---
  const add3 = add(norm2, ffnOutput);
  const { output: norm3, means: means3, variances: variances3 } = layerNorm(add3);

  // --- Stage 9: Output Projection ---
  const W_proj = randomMatrix(d_model, vocabSize, 341, -0.8, 0.8);
  const logits = matmul(norm3, W_proj);

  // --- Stage 10: Softmax & Prediction ---
  const probabilities = softmax(logits);

  return {
    tokens,
    vocab,
    tokenIds,
    W_embed,
    embeddings,
    pe,
    Z,
    Wq_self,
    Wk_self,
    Wv_self,
    Q_self,
    K_self,
    V_self,
    rawScores_self,
    scaledScores_self,
    maskedScores_self,
    attentionWeights_self,
    attentionOutput_self,
    add1,
    means1,
    variances1,
    norm1,
    Wq_cross,
    Wk_cross,
    Wv_cross,
    Q_cross,
    K_cross,
    V_cross,
    rawScores_cross,
    scaledScores_cross,
    attentionWeights_cross,
    attentionOutput_cross,
    add2,
    means2,
    variances2,
    norm2,
    W1,
    b1,
    H_raw,
    H_activated,
    W2,
    b2,
    ffnOutput,
    add3,
    means3,
    variances3,
    norm3,
    W_proj,
    logits,
    probabilities,
  };
}

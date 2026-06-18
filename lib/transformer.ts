import {
  randomMatrix,
  matmul,
  transpose,
  softmax,
  layerNorm,
  add,
  relu,
  scale,
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

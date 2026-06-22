/**
 * Seeded pseudo-random number generator using Mulberry32.
 * Output is deterministic based on the seed value.
 */
export function seededRandom(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generates a matrix of shape (rows, cols) with deterministic pseudo-random values.
 */
export function randomMatrix(
  rows: number,
  cols: number,
  seed: number,
  min = -0.5,
  max = 0.5
): number[][] {
  const rand = seededRandom(seed);
  const mat: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push(min + rand() * (max - min));
    }
    mat.push(row);
  }
  return mat;
}

/**
 * Standard matrix multiplication C = A * B.
 * A is (M x N), B is (N x P), result is (M x P).
 */
export function matmul(A: number[][], B: number[][]): number[][] {
  const M = A.length;
  const N = A[0].length;
  const P = B[0].length;
  const C: number[][] = [];
  
  for (let i = 0; i < M; i++) {
    const row: number[] = [];
    for (let j = 0; j < P; j++) {
      let sum = 0;
      for (let k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      row.push(sum);
    }
    C.push(row);
  }
  return C;
}

/**
 * Returns the transpose of matrix A.
 */
export function transpose(A: number[][]): number[][] {
  const M = A.length;
  const N = A[0].length;
  const T: number[][] = [];
  
  for (let j = 0; j < N; j++) {
    const row: number[] = [];
    for (let i = 0; i < M; i++) {
      row.push(A[i][j]);
    }
    T.push(row);
  }
  return T;
}

/**
 * Applies row-wise softmax with numerical stability.
 */
export function softmax(A: number[][]): number[][] {
  return A.map((row) => {
    const max = Math.max(...row);
    const exps = row.map((val) => Math.exp(val - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((val) => val / (sum || 1));
  });
}

/**
 * Standard Layer Normalization applied to each row.
 * Returns the normalized matrix, as well as intermediate means and variances.
 */
export function layerNorm(
  A: number[][],
  gamma?: number[],
  beta?: number[]
): { output: number[][]; means: number[]; variances: number[] } {
  const M = A.length;
  const N = A[0].length;
  const eps = 1e-5;

  const means: number[] = [];
  const variances: number[] = [];
  const output: number[][] = [];

  const g = gamma || Array(N).fill(1);
  const b = beta || Array(N).fill(0);

  for (let i = 0; i < M; i++) {
    const row = A[i];
    const sum = row.reduce((acc, val) => acc + val, 0);
    const mean = sum / N;
    means.push(mean);

    const sqSum = row.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
    const variance = sqSum / N;
    variances.push(variance);

    const normRow = row.map((val, colIdx) => {
      const normalized = (val - mean) / Math.sqrt(variance + eps);
      return normalized * g[colIdx] + b[colIdx];
    });
    output.push(normRow);
  }

  return { output, means, variances };
}

/**
 * Element-wise addition of two matrices of the same shape.
 */
export function add(A: number[][], B: number[][]): number[][] {
  return A.map((row, r) => row.map((val, c) => val + B[r][c]));
}

/**
 * Element-wise ReLU activation function.
 */
export function relu(A: number[][]): number[][] {
  return A.map((row) => row.map((val) => Math.max(0, val)));
}

/**
 * Element-wise scalar multiplication.
 */
export function scale(A: number[][], scalar: number): number[][] {
  return A.map((row) => row.map((val) => val * scalar));
}

/**
 * Applies a look-ahead mask to attention scores.
 * Sets positions above the diagonal (j > i) to -Infinity.
 */
export function applyMask(scores: number[][]): number[][] {
  return scores.map((row, r) =>
    row.map((val, c) => (c > r ? -Infinity : val))
  );
}

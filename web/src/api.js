/**
 * API service for communicating with FastAPI backend.
 * Falls back to demo data when backend is offline.
 */

const BASE = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api` : '/api'


async function post(path, formData) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body: formData })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function predictModel(modelName, file) {
  const fd = new FormData()
  fd.append('file', file)
  return post(`/predict/${modelName}`, fd)
}

export async function predictAll(file, maskFile = null) {
  const fd = new FormData()
  fd.append('file', file)
  if (maskFile) fd.append('mask', maskFile)
  return post('/predict/all', fd)
}

export async function getMetrics() {
  const res = await fetch(`${BASE}/metrics`)
  if (!res.ok) throw new Error('Metrics fetch failed')
  return res.json()
}

// ─── Demo / Mock Data ─────────────────────────────────────

// 1×1 transparent PNG — enough to satisfy canvas rendering in demo mode.
// The DemoMaskOverlay will draw the actual MRI + overlay itself.
const DEMO_MASK_B64 = null

export function demoSegResult() {
  return {
    model: 'unet',
    tensor_shape: [1, 1, 240, 240],
    logit_range: [-4.312, 6.118],
    output_type: 'segmentation_logits',
    dice: 0.7214,
    mask_overlay_b64: DEMO_MASK_B64,   // null → InferencePanel uses DemoMaskOverlay
    extra: { sigmoid_max: 0.9982, positive_voxels: 4821 },
  }
}

export function demoClsResult(prob = 0.942) {
  return {
    model: 'baseline_cls',
    tensor_shape: [1, 1],
    logit_range: [-0.06, 2.78],
    output_type: 'classification_logit',
    tumor_prob: prob,
    confidence_pct: +(Math.max(prob, 1 - prob) * 100).toFixed(2),
    extra: { sigmoid: prob },
  }
}

export function demoMtlResult() {
  return {
    model: 'standard_mtl',
    tensor_shape: [1, 1, 240, 240],
    logit_range: [-4.1, 5.9],
    output_type: 'multi_task_logits',
    dice: 0.7012,
    mask_overlay_b64: DEMO_MASK_B64,   // null → DemoMaskOverlay
    tumor_prob: 0.958,
    confidence_pct: 95.8,
    extra: { seg_sigmoid_max: 0.9976, cls_sigmoid: 0.958 },
  }
}

export function demoAllResults() {
  return [
    { model: 'unet', type: 'Segmentation', dice: 0.721, tumor_prob: null, confidence: null, confidence_pct: null },
    { model: 'baseline_cls', type: 'Classification', dice: null, tumor_prob: 0.942, confidence: 94.2, confidence_pct: 94.2 },
    { model: 'transfer_cls', type: 'Classification', dice: null, tumor_prob: 0.991, confidence: 99.1, confidence_pct: 99.1 },
    { model: 'standard_mtl', type: 'Multi-Task', dice: 0.701, tumor_prob: 0.952, confidence: 95.2, confidence_pct: 95.2 },
    { model: 'pcgrad_mtl', type: 'Multi-Task', dice: 0.732, tumor_prob: 0.988, confidence: 98.8, confidence_pct: 98.8 },
  ]
}

export const BENCHMARK_METRICS = [
  {
    rank: 1,
    model: 'PCGrad MTL',
    optimization: 'PCGrad + Uncertainty Weights',
    valAUC: 0.9921,
    valDice: 0.732,
    epochs: 20,
    conflict: 'Resolved (38.2%)',
    conflictColor: 'green',
  },
  {
    rank: 2,
    model: 'Standard MTL',
    optimization: 'Adam + Uncertainty Weights',
    valAUC: 0.9874,
    valDice: 0.701,
    epochs: 20,
    conflict: 'Unresolved',
    conflictColor: 'orange',
  },
  {
    rank: 3,
    model: 'Transfer Learning',
    optimization: 'Adam (Frozen Encoder)',
    valAUC: 0.9910,
    valDice: null,
    epochs: 25,
    conflict: 'N/A',
    conflictColor: 'muted',
  },
  {
    rank: 4,
    model: 'Baseline Classification',
    optimization: 'Adam',
    valAUC: 0.9862,
    valDice: null,
    epochs: 25,
    conflict: 'N/A',
    conflictColor: 'muted',
  },
  {
    rank: 5,
    model: 'UNet Segmentation',
    optimization: 'Adam + Focal+Dice Loss',
    valAUC: null,
    valDice: 0.721,
    epochs: 20,
    conflict: 'N/A',
    conflictColor: 'muted',
  },
]

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Trophy, Zap, BarChart2, Brain, Activity, Layers, CheckCircle, Upload
} from 'lucide-react'
import Dropzone from '../components/Dropzone'
import ProbChart from '../components/ProbChart'
import { DemoMaskOverlay } from '../components/MaskOverlay'
import { predictAll, demoAllResults, BENCHMARK_METRICS } from '../api'

/* ── Model metadata ── */
const MODEL_META = {
  unet:         { name: 'UNet Seg.',     icon: Brain,    color: 'var(--electric-blue)' },
  baseline_cls: { name: 'Baseline CLS', icon: BarChart2, color: 'var(--neon-cyan)'    },
  transfer_cls: { name: 'Transfer CLS', icon: Layers,    color: 'var(--neon-orange)'  },
  standard_mtl: { name: 'Standard MTL', icon: Activity,  color: 'var(--neon-purple)'  },
  pcgrad_mtl:   { name: 'PCGrad MTL',   icon: Zap,       color: 'var(--neon-green)'   },
}

/* ── Leaderboard ── */
function MetricCell({ value, isBest }) {
  if (value == null) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  return (
    <span className={`metric-cell${isBest ? ' best' : ''}`}>
      {typeof value === 'number' ? value.toFixed(3) : value}
    </span>
  )
}

const CONFLICT_COLORS = {
  green:  { color: 'var(--neon-green)',  bg: 'rgba(22,163,74,0.08)',  border: 'rgba(22,163,74,0.2)' },
  orange: { color: 'var(--neon-orange)', bg: 'rgba(234,88,12,0.08)',  border: 'rgba(234,88,12,0.2)' },
  muted:  { color: 'var(--text-muted)',  bg: 'rgba(0,0,0,0.03)',      border: 'rgba(0,0,0,0.08)'   },
}

function Leaderboard() {
  const bestAUC  = Math.max(...BENCHMARK_METRICS.filter(m => m.valAUC  != null).map(m => m.valAUC))
  const bestDice = Math.max(...BENCHMARK_METRICS.filter(m => m.valDice != null).map(m => m.valDice))

  return (
    <div className="card">
      <div className="card-header">
        <Trophy size={15} color="var(--neon-amber)" />
        <div>
          <div className="card-title">Historical Performance Leaderboard</div>
          <div className="card-subtitle">Pre-computed validation metrics from W&B training runs</div>
        </div>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        <div className="benchmark-table-wrap">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Model</th>
                <th>Optimization Strategy</th>
                <th>Val ROC-AUC ↑</th>
                <th>Val Dice ↑</th>
                <th>Epochs</th>
                <th>Gradient Conflict</th>
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_METRICS.map(m => {
                const cc = CONFLICT_COLORS[m.conflictColor] || CONFLICT_COLORS.muted
                return (
                  <tr key={m.rank} className={m.rank === 1 ? 'rank-1' : ''}>
                    <td>
                      <span style={{
                        width: 24, height: 24, borderRadius: '50%', display: 'inline-flex',
                        alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 800,
                        background: m.rank === 1 ? 'linear-gradient(135deg,#0284c7,#4f46e5)' : 'var(--bg-secondary)',
                        color: m.rank === 1 ? '#fff' : 'var(--text-muted)',
                        border: m.rank === 1 ? 'none' : '1px solid var(--border-subtle)',
                      }}>
                        {m.rank === 1 ? '🥇' : m.rank}
                      </span>
                    </td>
                    <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{m.model}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{m.optimization}</td>
                    <td><MetricCell value={m.valAUC}  isBest={m.valAUC  === bestAUC}  /></td>
                    <td><MetricCell value={m.valDice} isBest={m.valDice === bestDice} /></td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 12 }}>{m.epochs}</td>
                    <td>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 100,
                        fontSize: 11, fontWeight: 600,
                        color: cc.color, background: cc.bg, border: `1px solid ${cc.border}`,
                      }}>
                        {m.conflict}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

/* ── Results matrix row ── */
function ResultRow({ modelKey, r, imageUrl, isComplete }) {
  const meta = MODEL_META[modelKey] || { name: modelKey, color: 'var(--text-muted)' }
  const Icon = meta.icon || Brain
  const isRunning = !isComplete

  return (
    <tr>
      {/* Model */}
      <td>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: `color-mix(in srgb, ${meta.color} 12%, white)`,
            border: `1px solid color-mix(in srgb, ${meta.color} 25%, white)`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <Icon size={13} color={meta.color} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--text-primary)' }}>{meta.name}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {r.type === 'Segmentation' ? 'Seg only' : r.type === 'Classification' ? 'CLS only' : 'Seg + CLS'}
            </div>
          </div>
        </div>
      </td>

      {/* Type badge */}
      <td>
        <span className={`tag ${r.type === 'Multi-Task' ? 'purple' : r.type === 'Segmentation' ? 'blue' : 'orange'}`}
          style={{ fontSize: 10 }}>
          {r.type}
        </span>
      </td>

      {/* Status / Segmentation preview */}
      <td>
        {isRunning ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: 11 }}>
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
              <Zap size={12} color="var(--electric-blue)" />
            </motion.div>
            Running…
          </div>
        ) : (r.type === 'Segmentation' || r.type === 'Multi-Task') && imageUrl ? (
          <DemoMaskOverlay imageUrl={imageUrl} diceScore={r.dice} />
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>—</span>
        )}
      </td>

      {/* Classification output */}
      <td style={{ minWidth: 140 }}>
        {isRunning ? (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>—</span>
        ) : r.tumor_prob != null ? (
          <ProbChart tumorProb={r.tumor_prob} />
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>N/A</span>
        )}
      </td>

      {/* Metrics */}
      <td>
        {!isComplete ? (
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {r.dice != null && (
              <span className="tag green" style={{ fontSize: 10 }}>Dice: {r.dice.toFixed(3)}</span>
            )}
            {r.confidence_pct != null && (
              <span className="tag blue" style={{ fontSize: 10 }}>Conf: {Number(r.confidence_pct).toFixed(1)}%</span>
            )}
          </div>
        )}
      </td>
    </tr>
  )
}

/* ── Main Page ── */
export default function BenchmarkPage({ backendStatus }) {
  const [file,      setFile]      = useState(null)
  const [maskFile,  setMaskFile]  = useState(null)
  const [imageUrl,  setImageUrl]  = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [results,   setResults]   = useState([])
  const [completed, setCompleted] = useState(new Set())
  const [progress,  setProgress]  = useState(0)

  const handleMRI  = (f) => { setFile(f);     setImageUrl(f?.type.startsWith('image/') ? URL.createObjectURL(f) : null); resetResults() }
  const handleMask = (f) => { setMaskFile(f); resetResults() }
  const resetResults = () => { setResults([]); setCompleted(new Set()); setProgress(0) }

  const allModels = Object.keys(MODEL_META)

  const launch = async () => {
    if (!file) return
    setIsRunning(true)
    resetResults()

    try {
      let allResults
      if (backendStatus === 'online') {
        allResults = await predictAll(file, maskFile)
      } else {
        // Demo: stagger each model completion for visual effect
        allResults = demoAllResults()
        for (let i = 0; i < allModels.length; i++) {
          await new Promise(r => setTimeout(r, 500 + Math.random() * 400))
          setCompleted(prev => new Set([...prev, allModels[i]]))
          setProgress(Math.round(((i + 1) / allModels.length) * 100))
        }
        setResults(allResults)
        setIsRunning(false)
        return
      }
      setResults(allResults)
      allModels.forEach(m => setCompleted(prev => new Set([...prev, m])))
      setProgress(100)
    } catch {
      const demo = demoAllResults()
      for (let i = 0; i < allModels.length; i++) {
        await new Promise(r => setTimeout(r, 450 + Math.random() * 350))
        setCompleted(prev => new Set([...prev, allModels[i]]))
        setProgress(Math.round(((i + 1) / allModels.length) * 100))
      }
      setResults(demo)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap" style={{ background: 'rgba(217,119,6,0.08)', borderColor: 'rgba(217,119,6,0.25)' }}>
            <Trophy size={22} color="var(--neon-amber)" />
          </div>
          <div>
            <h1 className="page-title">Model Benchmark &amp; Live Challenge Suite</h1>
            <p className="page-subtitle">Compare all 5 architectures simultaneously on any MRI slice</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <span className="tag amber">5 Models</span>
          <span className="tag green">Live Inference</span>
          <span className="tag blue">Side-by-Side</span>
        </div>
      </div>

      <div className="page-body" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* ── Leaderboard ── */}
        <Leaderboard />

        {/* ── Live Challenge ── */}
        <div className="card">
          <div className="card-header">
            <Zap size={15} color="var(--neon-purple)" />
            <div>
              <div className="card-title">Live Multi-Model Inference Challenge</div>
              <div className="card-subtitle">Upload an MRI slice to benchmark all 5 models simultaneously</div>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                  MRI Slice <span style={{ color: 'var(--neon-red)' }}>*</span>
                </div>
                <Dropzone onFile={handleMRI} label="MRI slice" />
              </div>
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                  Ground Truth Mask <span style={{ color: 'var(--text-muted)' }}>(optional)</span>
                </div>
                <Dropzone onFile={handleMask} label="GT mask" />
              </div>
            </div>

            <button
              className="run-btn"
              onClick={launch}
              disabled={!file || isRunning}
              id="benchmark-run-btn"
              style={{ background: 'linear-gradient(135deg, #d97706 0%, #7c3aed 100%)' }}
            >
              <Zap size={15} />
              {isRunning ? 'Running All 5 Models…' : 'Launch Multi-Model Challenge'}
            </button>

            {/* Progress */}
            <AnimatePresence>
              {(isRunning || progress > 0) && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ marginTop: 16, overflow: 'hidden' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      {isRunning ? 'Running forward passes…' : 'Complete'}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--electric-blue)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                      {progress}%
                    </span>
                  </div>
                  <div className="progress-bar-wrap">
                    <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                  </div>

                  {/* Per-model pills */}
                  <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
                    {allModels.map(m => {
                      const done = completed.has(m)
                      const meta = MODEL_META[m]
                      const Icon = meta.icon
                      return (
                        <motion.div key={m}
                          animate={{ scale: done ? [1, 1.06, 1] : 1 }}
                          transition={{ duration: 0.3 }}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 5,
                            padding: '4px 10px', borderRadius: 100,
                            background: done ? 'rgba(22,163,74,0.08)' : 'var(--bg-secondary)',
                            border: `1px solid ${done ? 'rgba(22,163,74,0.25)' : 'var(--border-subtle)'}`,
                            fontSize: 11, fontWeight: 600,
                            color: done ? 'var(--neon-green)' : 'var(--text-muted)',
                          }}
                        >
                          {done ? <CheckCircle size={11} /> : (
                            <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
                              <Zap size={11} />
                            </motion.div>
                          )}
                          {meta.name}
                        </motion.div>
                      )
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* ── Results Matrix ── */}
        <AnimatePresence>
          {(results.length > 0 || isRunning) && (
            <motion.div
              className="card"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="card-header">
                <BarChart2 size={15} color="var(--electric-blue)" />
                <div>
                  <div className="card-title">Results Comparison Matrix</div>
                  <div className="card-subtitle">All 5 models compared on your uploaded slice</div>
                </div>
              </div>
              <div className="card-body" style={{ padding: 0, overflowX: 'auto' }}>
                <table className="challenge-results-table" style={{ minWidth: 680 }}>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Output Type</th>
                      <th>Segmentation</th>
                      <th>Classification</th>
                      <th>Metrics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allModels.map(modelKey => {
                      const r = results.find(x => x.model === modelKey) || {
                        model: modelKey,
                        type: modelKey === 'unet' ? 'Segmentation' : modelKey.includes('cls') ? 'Classification' : 'Multi-Task',
                        dice: null, tumor_prob: null, confidence: null, confidence_pct: null,
                      }
                      return (
                        <ResultRow
                          key={modelKey}
                          modelKey={modelKey}
                          r={r}
                          imageUrl={imageUrl}
                          isComplete={completed.has(modelKey)}
                        />
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </>
  )
}

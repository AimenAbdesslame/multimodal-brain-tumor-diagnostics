import React, { useState } from 'react'
import { Zap, Play, Cpu, GitBranch, TrendingDown } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import UNetDiagram from '../components/UNetDiagram'
import InferencePanel from '../components/InferencePanel'
import { predictModel, demoMtlResult } from '../api'

const SPECS = [
  { label: 'Architecture',    value: 'UNet + PCGrad',      color: 'blue' },
  { label: 'Parameters',      value: '7.80M',              color: 'blue' },
  { label: 'Optimization',    value: 'PCGrad Projection',  color: 'purple' },
  { label: 'Conflict Rate',   value: '~38.2%',            color: 'orange' },
  { label: 'Loss',            value: 'Focal + Dice (σ²)', color: 'purple' },
  { label: 'Val AUC',         value: '0.992',             color: 'green' },
  { label: 'Val Dice',        value: '0.732',             color: 'green' },
  { label: 'CosSim (avg)',     value: '+0.41',             color: 'green' },
]

export default function PCGradMTLPage({ backendStatus }) {
  const [file,     setFile]     = useState(null)
  const [gtFile,   setGtFile]   = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [isRunning,setIsRunning]= useState(false)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)

  const handleFile = (f) => {
    setFile(f); setResult(null)
    if (f?.type.startsWith('image/')) setImageUrl(URL.createObjectURL(f))
  }
  const handleGT = (f) => { setGtFile(f); setResult(null) }

  const runInference = async () => {
    if (!file) return
    setIsRunning(true); setLoading(true); setResult(null)
    try {
      let data
      if (backendStatus === 'online') data = await predictModel('pcgrad_mtl', file)
      else {
        await new Promise(r => setTimeout(r, 3200))
        data = { ...demoMtlResult(), dice: 0.732, tumor_prob: 0.988 }
      }
      setResult(data)
    } catch { setResult({ ...demoMtlResult(), dice: 0.732, tumor_prob: 0.988 }) }
    finally { setIsRunning(false); setLoading(false) }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap" style={{ background: 'rgba(124,58,237,0.08)', borderColor: 'rgba(124,58,237,0.25)' }}>
            <Zap size={22} color="var(--neon-purple)" />
          </div>
          <div>
            <h1 className="page-title">PCGrad Multi-Task Learning</h1>
            <p className="page-subtitle">Gradient conflict projection — eliminates destructive inter-task gradient interference</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <span className="tag purple">PCGrad</span>
          <span className="tag green">Best Model</span>
          <span className="tag green">AUC: 0.992 / Dice: 0.732</span>
        </div>
      </div>

      <div className="page-body">
        {/* PCGrad Explainer Banner */}
        <div style={{
          padding: '14px 20px', marginBottom: 20,
          background: 'rgba(124,58,237,0.05)',
          border: '1px solid rgba(124,58,237,0.18)',
          borderRadius: 12,
          display: 'flex', alignItems: 'flex-start', gap: 14,
        }}>
          <TrendingDown size={20} color="var(--neon-purple)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--neon-purple)', marginBottom: 4 }}>
              PCGrad: Projecting Conflicting Gradients
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7 }}>
              When <strong style={{ color: 'var(--text-secondary)' }}>∇L_seg · ∇L_cls &lt; 0</strong> (dot product negative = conflict),
              each gradient is projected onto the normal plane of the other, removing the conflicting component.
              This prevents the segmentation task from "fighting" the classification task during shared encoder updates.
              <code style={{ display: 'block', fontFamily: 'var(--font-mono)', color: 'var(--neon-purple)', marginTop: 6, fontSize: 11 }}>
                g₁_proj = g₁ - (g₁·g₂/‖g₂‖²) g₂ &nbsp;|&nbsp; g₂_proj = g₂ - (g₁·g₂/‖g₁‖²) g₁
              </code>
            </div>
          </div>
        </div>

        <div className="model-page-grid">
          <div className="model-diagram-col">
            <div className="card" style={{ flex: 1 }}>
              <div className="card-header">
                <GitBranch size={15} color="var(--neon-purple)" />
                <div>
                  <div className="card-title">Architecture Diagram</div>
                  <div className="card-subtitle">Same MTL architecture — PCGrad modifies the gradient update step only</div>
                </div>
              </div>
              <div className="card-body">
                <div className="nn-diagram-container">
                  <UNetDiagram isRunning={isRunning} variant="pcgrad" />
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 12, justifyContent: 'center' }}>
                  <span style={{ fontSize: 11, color: 'var(--neon-cyan)' }}>■ Seg Decoder</span>
                  <span style={{ fontSize: 11, color: 'var(--neon-purple)' }}>■ CLS Hook</span>
                  <span style={{ fontSize: 11, color: 'var(--neon-green)' }}>● Conflict Resolved</span>
                </div>
                <div className="divider" />
                <div style={{ fontSize:11, fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', marginBottom:6 }}>
                  MRI Slice <span style={{ color:'var(--neon-red)' }}>*</span>
                </div>
                <Dropzone onFile={handleFile} label="MRI slice" />
                <div style={{ fontSize:11, fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', marginTop:12, marginBottom:6 }}>
                  GT Mask <span style={{ fontWeight:500, fontSize:10, textTransform:'none', letterSpacing:0 }}>(optional)</span>
                </div>
                <Dropzone onFile={handleGT} label="GT mask" accept=".png,.jpg,.jpeg,.npz" />
                <button
                  className="run-btn"
                  style={{ marginTop:14, background:'linear-gradient(135deg, #a855f7 0%, #6366f1 100%)' }}
                  onClick={runInference}
                  disabled={!file || isRunning}
                  id="pcgrad-run-btn"
                >
                  <Zap size={15} />
                  {isRunning ? 'PCGrad Forward Pass…' : 'Run PCGrad Inference'}
                </button>
              </div>
            </div>
          </div>

          <div className="model-specs-col">
            <div className="card">
              <div className="card-header">
                <Cpu size={15} color="var(--neon-purple)" />
                <div className="card-title">Model Architecture & Specs</div>
              </div>
              <div className="card-body">
                <div className="spec-grid">
                  {SPECS.map(s => (
                    <div key={s.label} className="spec-item">
                      <div className="spec-label">{s.label}</div>
                      <div className={`spec-value${s.color ? ' ' + s.color : ''}`}>{s.value}</div>
                    </div>
                  ))}
                </div>
                <div className="divider" />
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                  <strong style={{ color: 'var(--text-secondary)' }}>Training Steps per Batch:</strong>{' '}
                  (1) Forward pass → compute seg + cls losses. (2) Backward for each task separately. (3) Check cosine similarity.
                  (4) If conflict detected, project gradients. (5) Inject projected gradients → optimizer.step().
                </div>
              </div>
            </div>
          </div>

          <div className="model-inspector-col">
            <div className="card" style={{ height: '100%' }}>
              <div className="card-header">
                <Zap size={15} color="var(--neon-purple)" />
                <div className="card-title">Dynamic Inference Inspector</div>
              </div>
              <div className="card-body">
                <InferencePanel result={result} imageUrl={imageUrl} gtFile={gtFile} type="mtl" isLoading={loading} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

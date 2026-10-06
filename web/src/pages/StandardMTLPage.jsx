import React, { useState } from 'react'
import { Activity, Play, Cpu, GitBranch } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import UNetDiagram from '../components/UNetDiagram'
import InferencePanel from '../components/InferencePanel'
import { predictModel, demoMtlResult } from '../api'

const SPECS = [
  { label: 'Architecture',     value: 'UNet + MTL',         color: 'blue' },
  { label: 'Parameters',       value: '7.80M',              color: 'blue' },
  { label: 'Seg Head',         value: 'ConvTranspose2d',    color: '' },
  { label: 'CLS Head',         value: 'GAP → MLP',         color: '' },
  { label: 'Loss',             value: 'Focal + Dice (σ²)',  color: 'purple' },
  { label: 'Optimizer',        value: 'Adam (1e-4)',         color: '' },
  { label: 'Val AUC',          value: '0.987',              color: 'green' },
  { label: 'Val Dice',         value: '0.701',              color: 'green' },
]

export default function StandardMTLPage({ backendStatus }) {
  const [file,     setFile]     = useState(null)
  const [gtFile,   setGtFile]   = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [isRunning,setIsRunning]= useState(false)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)

  const handleFile = (f) => {
    setFile(f); setResult(null)
    setImageUrl(f?.type.startsWith('image/') ? URL.createObjectURL(f) : null)
  }
  const handleGT = (f) => { setGtFile(f); setResult(null) }

  const runInference = async () => {
    if (!file) return
    setIsRunning(true); setLoading(true); setResult(null)
    try {
      let data
      if (backendStatus === 'online') data = await predictModel('standard_mtl', file)
      else { await new Promise(r => setTimeout(r, 3000)); data = demoMtlResult() }
      setResult(data)
    } catch { setResult(demoMtlResult()) }
    finally { setIsRunning(false); setLoading(false) }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap"><Activity size={22} /></div>
          <div>
            <h1 className="page-title">Standard Multi-Task Learning</h1>
            <p className="page-subtitle">Shared UNet encoder with dual segmentation + classification heads</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <span className="tag blue">Multi-Task</span>
          <span className="tag purple">Uncertainty Weighting</span>
          <span className="tag green">AUC: 0.987 / Dice: 0.701</span>
        </div>
      </div>

      <div className="page-body">
        <div className="model-page-grid">
          <div className="model-diagram-col">
            <div className="card" style={{ flex: 1 }}>
              <div className="card-header">
                <GitBranch size={15} color="var(--electric-blue)" />
                <div>
                  <div className="card-title">Architecture Diagram</div>
                  <div className="card-subtitle">Shared UNet encoder → Segmentation decoder + CLS bottleneck hook</div>
                </div>
              </div>
              <div className="card-body">
                <div className="nn-diagram-container">
                  <UNetDiagram isRunning={isRunning} variant="mtl" />
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 12, justifyContent: 'center' }}>
                  <span style={{ fontSize: 11, color: 'var(--neon-cyan)' }}>■ Seg Head (Decoder)</span>
                  <span style={{ fontSize: 11, color: 'var(--neon-purple)' }}>■ CLS Head (Hook)</span>
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
                <button className="run-btn" style={{ marginTop:14 }} onClick={runInference} disabled={!file || isRunning} id="mtl-run-btn">
                  <Play size={15} />
                  {isRunning ? 'Multi-Task Forward Pass…' : 'Run Inference'}
                </button>
              </div>
            </div>
          </div>

          <div className="model-specs-col">
            <div className="card">
              <div className="card-header">
                <Cpu size={15} color="var(--electric-blue)" />
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
                  <strong style={{ color: 'var(--text-secondary)' }}>Uncertainty Weighting:</strong>{' '}
                  Task-specific learnable log-variance parameters (σ²) adaptively balance segmentation
                  and classification losses:
                  <code style={{ display: 'block', fontFamily: 'var(--font-mono)', color: 'var(--neon-orange)', marginTop: 6, fontSize: 11 }}>
                    L = Σ (1/2σᵢ²)Lᵢ + log(σᵢ)
                  </code>
                </div>
              </div>
            </div>
          </div>

          <div className="model-inspector-col">
            <div className="card" style={{ height: '100%' }}>
              <div className="card-header">
                <Activity size={15} color="var(--electric-blue)" />
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

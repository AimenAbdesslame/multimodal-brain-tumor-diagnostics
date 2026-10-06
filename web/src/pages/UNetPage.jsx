import React, { useState } from 'react'
import { Brain, Play, Cpu, GitBranch } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import UNetDiagram from '../components/UNetDiagram'
import InferencePanel from '../components/InferencePanel'
import { predictModel, demoSegResult } from '../api'

const SPECS = [
  { label: 'Architecture',  value: 'UNet (3-level)', color: 'blue'   },
  { label: 'Parameters',    value: '7.76M',          color: 'blue'   },
  { label: 'Input Shape',   value: '1×240×240',      color: ''       },
  { label: 'Output Shape',  value: '1×240×240',      color: ''       },
  { label: 'Loss Function', value: 'Focal + Dice',   color: 'purple' },
  { label: 'Optimizer',     value: 'Adam (1e-4)',     color: ''       },
  { label: 'Batch Size',    value: '16',             color: ''       },
  { label: 'Best Val Dice', value: '0.7214',         color: 'green'  },
]

export default function UNetPage({ backendStatus }) {
  const [file,     setFile]     = useState(null)
  const [gtFile,   setGtFile]   = useState(null)   // ground truth mask (optional)
  const [imageUrl, setImageUrl] = useState(null)
  const [isRunning,setIsRunning]= useState(false)
  const [result,   setResult]   = useState(null)
  const [loading,  setLoading]  = useState(false)

  const handleFile = (f) => {
    setFile(f)
    setResult(null)
    setImageUrl(f?.type.startsWith('image/') ? URL.createObjectURL(f) : null)
  }

  const handleGT = (f) => {
    setGtFile(f)
    setResult(null)
  }

  const runInference = async () => {
    if (!file) return
    setIsRunning(true); setLoading(true); setResult(null)
    try {
      const data = backendStatus === 'online'
        ? await predictModel('unet', file)
        : await new Promise(r => setTimeout(r, 2600)).then(() => demoSegResult())
      setResult(data)
    } catch {
      setResult(demoSegResult())
    } finally {
      setIsRunning(false); setLoading(false)
    }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap">
            <Brain size={22} />
          </div>
          <div>
            <h1 className="page-title">UNet Segmentation</h1>
            <p className="page-subtitle">
              Encoder-decoder with skip connections for pixel-wise tumor mask prediction
            </p>
          </div>
        </div>
        <div style={{ display:'flex', gap:8, marginTop:12 }}>
          <span className="tag blue">Segmentation</span>
          <span className="tag purple">Focal + Dice Loss</span>
          <span className="tag green">Val Dice: 0.7214</span>
        </div>
      </div>

      <div className="page-body">
        <div className="model-page-grid">

          {/* ── Left: Architecture Diagram + Upload Controls ── */}
          <div className="model-diagram-col">
            <div className="card" style={{ flex:1 }}>
              <div className="card-header">
                <GitBranch size={15} color="var(--electric-blue)" />
                <div>
                  <div className="card-title">Interactive Architecture Diagram</div>
                  <div className="card-subtitle">U-shaped encoder-decoder — animated forward pass on Run Inference</div>
                </div>
              </div>
              <div className="card-body">
                <div className="nn-diagram-container">
                  <UNetDiagram isRunning={isRunning} variant="unet" />
                </div>

                <div className="divider" />

                {/* MRI Upload */}
                <div style={{ fontSize:11, fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', marginBottom:6 }}>
                  MRI Slice <span style={{ color:'var(--neon-red)' }}>*</span>
                </div>
                <Dropzone onFile={handleFile} label="MRI slice" accept=".png,.jpg,.jpeg,.npz" />

                {/* Ground Truth Upload (optional) */}
                <div style={{ fontSize:11, fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', marginTop:12, marginBottom:6 }}>
                  Ground Truth Mask{' '}
                  <span style={{ fontWeight:500, fontSize:10, color:'var(--text-muted)', textTransform:'none', letterSpacing:'0' }}>
                    (optional — enables real GT panel)
                  </span>
                </div>
                <Dropzone onFile={handleGT} label="GT mask" accept=".png,.jpg,.jpeg,.npz" />

                <button
                  className="run-btn"
                  style={{ marginTop:14 }}
                  onClick={runInference}
                  disabled={!file || isRunning}
                  id="unet-run-btn"
                >
                  <Play size={15} />
                  {isRunning ? 'Running Forward Pass…' : 'Run Inference'}
                </button>
              </div>
            </div>
          </div>

          {/* ── Right top: Specs ── */}
          <div className="model-specs-col">
            <div className="card">
              <div className="card-header">
                <Cpu size={15} color="var(--electric-blue)" />
                <div className="card-title">Model Architecture &amp; Specs</div>
              </div>
              <div className="card-body">
                <div className="spec-grid">
                  {SPECS.map(s => (
                    <div key={s.label} className="spec-item">
                      <div className="spec-label">{s.label}</div>
                      <div className={`spec-value${s.color ? ' '+s.color : ''}`}>{s.value}</div>
                    </div>
                  ))}
                </div>
                <div className="divider" />
                <div style={{ fontSize:12, color:'var(--text-muted)', lineHeight:1.75 }}>
                  <strong style={{ color:'var(--text-secondary)' }}>Architecture:</strong>{' '}
                  3 encoder blocks (DoubleConv + MaxPool ↓), 512-ch bottleneck,
                  3 decoder blocks (ConvTranspose2d ↑) with concatenated skip connections.
                  Final 1×1 Conv → binary logit → sigmoid threshold 0.5.
                </div>
                <div className="divider" />
                {/* Notebook visualization note */}
                <div style={{
                  padding:'10px 13px',
                  background:'rgba(2,132,199,0.05)',
                  border:'1px solid rgba(2,132,199,0.15)',
                  borderRadius:8,
                  fontSize:11, color:'var(--text-muted)', lineHeight:1.7,
                }}>
                  📊 <strong style={{ color:'var(--text-secondary)' }}>Visualization</strong> follows
                  notebook <code style={{ fontFamily:'var(--font-mono)' }}>01_exp1_segmentation_baseline.ipynb</code>:
                  4-panel layout — Input MRI · GT (Greens) · Prediction (Reds) · Combined overlay.
                </div>
              </div>
            </div>
          </div>

          {/* ── Right bottom: Inference Inspector ── */}
          <div className="model-inspector-col">
            <div className="card" style={{ height:'100%' }}>
              <div className="card-header">
                <Brain size={15} color="var(--electric-blue)" />
                <div className="card-title">Segmentation Inspector</div>
              </div>
              <div className="card-body">
                <InferencePanel
                  result={result}
                  imageUrl={imageUrl}
                  gtFile={gtFile}
                  type="segmentation"
                  isLoading={loading}
                />
              </div>
            </div>
          </div>

        </div>
      </div>
    </>
  )
}

import React, { useState } from 'react'
import { Layers, Play, Cpu, GitBranch, Lock } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import ClassificationDiagram from '../components/ClassificationDiagram'
import InferencePanel from '../components/InferencePanel'
import { predictModel, demoClsResult } from '../api'

const SPECS = [
  { label: 'Encoder',        value: 'Frozen UNet',    color: 'blue' },
  { label: 'Parameters',     value: '7.78M (1.28M t.)', color: 'blue' },
  { label: 'Input Shape',    value: '1×240×240',      color: '' },
  { label: 'Output',         value: 'Sigmoid Logit',  color: '' },
  { label: 'Strategy',       value: 'Frozen Encoder', color: 'purple' },
  { label: 'Optimizer',      value: 'Adam (1e-4)',     color: '' },
  { label: 'Dropout',        value: '0.4',            color: '' },
  { label: 'Best Val AUC',   value: '0.991',          color: 'green' },
]

export default function TransferLearningPage({ backendStatus }) {
  const [file, setFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFile = (f) => {
    setFile(f); setResult(null)
    if (f?.type.startsWith('image/')) setImageUrl(URL.createObjectURL(f))
  }

  const runInference = async () => {
    if (!file) return
    setIsRunning(true); setLoading(true); setResult(null)
    try {
      let data
      if (backendStatus === 'online') data = await predictModel('transfer_cls', file)
      else { await new Promise(r => setTimeout(r, 2400)); data = demoClsResult(0.991) }
      setResult(data)
    } catch { setResult(demoClsResult(0.991)) }
    finally { setIsRunning(false); setLoading(false) }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap"><Layers size={22} /></div>
          <div>
            <h1 className="page-title">Transfer Learning Classification</h1>
            <p className="page-subtitle">UNet pre-trained encoder (frozen) + fine-tuned MLP classification head</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <span className="tag blue">Transfer Learning</span>
          <span className="tag purple">Frozen Encoder</span>
          <span className="tag green">Val AUC: 0.991</span>
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
                  <div className="card-subtitle">
                    <Lock size={10} style={{ display: 'inline', marginRight: 4 }} />
                    Frozen UNet encoder weights → trainable MLP head
                  </div>
                </div>
              </div>
              <div className="card-body">
                {/* Frozen indicator */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '8px 12px', marginBottom: 12,
                  background: 'rgba(251,146,60,0.06)', border: '1px solid rgba(251,146,60,0.2)',
                  borderRadius: 8, fontSize: 12, color: 'var(--neon-orange)'
                }}>
                  <Lock size={12} /> Encoder weights frozen — only MLP head is trained
                </div>
                <div className="nn-diagram-container">
                  <ClassificationDiagram isRunning={isRunning} frozen={true} />
                </div>
                <div className="divider" />
                <Dropzone onFile={handleFile} label="MRI Slice" />
                <button className="run-btn" style={{ marginTop: 12 }} onClick={runInference} disabled={!file || isRunning} id="tl-run-btn">
                  <Play size={15} />
                  {isRunning ? 'Classifying…' : 'Run Inference'}
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
                  <strong style={{ color: 'var(--text-secondary)' }}>Transfer Strategy:</strong>{' '}
                  The UNet encoder (enc1–enc3 + bottleneck) is loaded from a pre-trained segmentation checkpoint
                  and frozen (<code style={{ fontFamily: 'var(--font-mono)', color: 'var(--neon-orange)' }}>requires_grad=False</code>).
                  Only the MLP classification head is trained, reducing GPU memory and training time significantly.
                </div>
              </div>
            </div>
          </div>

          <div className="model-inspector-col">
            <div className="card" style={{ height: '100%' }}>
              <div className="card-header">
                <Layers size={15} color="var(--electric-blue)" />
                <div className="card-title">Dynamic Inference Inspector</div>
              </div>
              <div className="card-body">
                <InferencePanel result={result} imageUrl={imageUrl} type="transfer" isLoading={loading} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

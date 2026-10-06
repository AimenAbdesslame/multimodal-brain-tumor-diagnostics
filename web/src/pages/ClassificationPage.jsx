import React, { useState } from 'react'
import { BarChart2, Play, Cpu, GitBranch } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import ClassificationDiagram from '../components/ClassificationDiagram'
import InferencePanel from '../components/InferencePanel'
import { predictModel, demoClsResult } from '../api'

const SPECS = [
  { label: 'Architecture',    value: 'UNet + CLS Head',  color: 'blue' },
  { label: 'Parameters',      value: '7.78M',            color: 'blue' },
  { label: 'Input Shape',     value: '1×240×240',        color: '' },
  { label: 'Output',          value: 'Sigmoid Logit',    color: '' },
  { label: 'Loss Function',   value: 'Binary Focal',     color: 'purple' },
  { label: 'Optimizer',       value: 'Adam (1e-4)',       color: '' },
  { label: 'MLP Hidden',      value: '512 → 128 → 1',   color: '' },
  { label: 'Best Val AUC',    value: '0.986',            color: 'green' },
]

export default function ClassificationPage({ backendStatus }) {
  const [file, setFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFile = (f) => {
    setFile(f)
    setResult(null)
    if (f?.type.startsWith('image/')) setImageUrl(URL.createObjectURL(f))
  }

  const runInference = async () => {
    if (!file) return
    setIsRunning(true)
    setLoading(true)
    setResult(null)
    try {
      let data
      if (backendStatus === 'online') data = await predictModel('baseline_cls', file)
      else { await new Promise(r => setTimeout(r, 2400)); data = demoClsResult(0.942) }
      setResult(data)
    } catch { setResult(demoClsResult(0.942)) }
    finally { setIsRunning(false); setLoading(false) }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title-area">
          <div className="page-icon-wrap"><BarChart2 size={22} /></div>
          <div>
            <h1 className="page-title">Standard Classification</h1>
            <p className="page-subtitle">Custom CNN with UNet backbone + MLP head via bottleneck hook</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <span className="tag blue">Classification</span>
          <span className="tag purple">Binary Focal Loss</span>
          <span className="tag green">Val AUC: 0.986</span>
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
                  <div className="card-subtitle">UNet encoder → Bottleneck → GAP → MLP classifier</div>
                </div>
              </div>
              <div className="card-body">
                <div className="nn-diagram-container">
                  <ClassificationDiagram isRunning={isRunning} />
                </div>
                <div className="divider" />
                <Dropzone onFile={handleFile} label="MRI Slice" />
                <button className="run-btn" style={{ marginTop: 12 }} onClick={runInference} disabled={!file || isRunning} id="cls-run-btn">
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
                  <strong style={{ color: 'var(--text-secondary)' }}>Hook Mechanism:</strong>{' '}
                  A forward hook is registered on the UNet's bottleneck layer to capture 512-channel feature maps.
                  Global Average Pooling reduces the spatial dimensions before the MLP classification head.
                </div>
              </div>
            </div>
          </div>

          <div className="model-inspector-col">
            <div className="card" style={{ height: '100%' }}>
              <div className="card-header">
                <BarChart2 size={15} color="var(--electric-blue)" />
                <div className="card-title">Dynamic Inference Inspector</div>
              </div>
              <div className="card-body">
                <InferencePanel result={result} imageUrl={imageUrl} type="classification" isLoading={loading} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

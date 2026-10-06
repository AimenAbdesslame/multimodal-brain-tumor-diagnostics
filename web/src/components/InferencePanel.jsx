import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Cpu, Code2, ScanLine } from 'lucide-react'
import ProbChart from './ProbChart'
import { SegmentationViz } from './MaskOverlay'

function TensorBlock({ shape, logitRange, outputType, extra = {} }) {
  return (
    <div className="code-block">
      <div><span className="key"># Output Tensor Inspection</span></div>
      <div>
        <span className="key">shape       = </span>
        <span className="val">{JSON.stringify(shape)}</span>
      </div>
      <div>
        <span className="key">dtype       = </span>
        <span className="str">torch.float32</span>
      </div>
      <div>
        <span className="key">logit_range = </span>
        <span className="num">
          [{Array.isArray(logitRange)
            ? logitRange.map(v => Number(v).toFixed(3)).join(', ')
            : '—'}]
        </span>
      </div>
      <div>
        <span className="key">output_type = </span>
        <span className="str">"{outputType}"</span>
      </div>
      {Object.entries(extra).map(([k, v]) => (
        <div key={k}>
          <span className="key">{k.padEnd(11)} = </span>
          <span className="num">
            {typeof v === 'number' ? Number(v).toFixed(4) : JSON.stringify(v)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function InferencePanel({
  result,
  imageUrl,
  gtFile = null,          // File: uploaded ground truth mask
  type = 'segmentation',
  isLoading,
}) {
  const [expanded, setExpanded] = useState(false)

  /* ── Loading ── */
  if (isLoading) {
    return (
      <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:14, padding:32 }}>
        <motion.div animate={{ rotate:360 }} transition={{ duration:1.1, repeat:Infinity, ease:'linear' }}>
          <Cpu size={28} color="var(--electric-blue)" />
        </motion.div>
        <span style={{ fontSize:13, color:'var(--text-muted)', fontWeight:500 }}>Running inference…</span>
        <div className="progress-bar-wrap" style={{ width:'100%' }}>
          <motion.div
            className="progress-bar-fill"
            animate={{ width:['8%','82%','8%'] }}
            transition={{ duration:2, repeat:Infinity, ease:'easeInOut' }}
          />
        </div>
      </div>
    )
  }

  /* ── Empty ── */
  if (!result) {
    return (
      <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:10, padding:'36px 20px', opacity:0.4 }}>
        <ScanLine size={30} color="var(--text-muted)" />
        <p style={{ fontSize:13, color:'var(--text-muted)', textAlign:'center' }}>
          Upload an MRI slice and click <strong>Run Inference</strong> to see results
        </p>
      </div>
    )
  }

  const hasSeg = type === 'segmentation' || type === 'mtl'
  const hasCls = type === 'classification' || type === 'transfer' || type === 'mtl'

  return (
    <div className="animate-fade" style={{ display:'flex', flexDirection:'column', gap:16 }}>

      {/* Raw tensor toggle */}
      <div>
        <button
          id="tensor-toggle-btn"
          onClick={() => setExpanded(!expanded)}
          style={{
            display:'flex', alignItems:'center', gap:6,
            background:'none', border:'none', cursor:'pointer',
            color:'var(--text-muted)', fontSize:12, fontWeight:600, padding:0, marginBottom:8,
          }}
        >
          <Code2 size={13} />
          Raw Tensor Output
          {expanded ? <ChevronUp size={12}/> : <ChevronDown size={12}/>}
        </button>
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height:0, opacity:0 }}
              animate={{ height:'auto', opacity:1 }}
              exit={{ height:0, opacity:0 }}
              transition={{ duration:0.2 }}
              style={{ overflow:'hidden' }}
            >
              <TensorBlock
                shape={result.tensor_shape}
                logitRange={result.logit_range}
                outputType={result.output_type}
                extra={result.extra || {}}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="divider" />

      {/* ══ SEGMENTATION — 4-panel notebook layout ══ */}
      {hasSeg && imageUrl && (
        <div>
          <div style={{
            fontSize:12, fontWeight:700, color:'var(--text-secondary)',
            marginBottom:10, display:'flex', alignItems:'center', gap:6,
          }}>
            <div style={{ width:10, height:10, borderRadius:2, background:'var(--electric-blue)' }} />
            Segmentation Output
          </div>

          {/* 4-panel visualization matching notebook cell 6 */}
          <SegmentationViz
            imageUrl={imageUrl}
            diceScore={result.dice}
            maskB64={result.mask_overlay_b64 || null}
            gtFile={gtFile}
          />

          {/* Dice score badge */}
          {result.dice != null && (
            <div style={{
              display:'flex', justifyContent:'space-between', alignItems:'center',
              marginTop:12, padding:'9px 14px',
              background:'rgba(22,163,74,0.06)',
              border:'1px solid rgba(22,163,74,0.2)',
              borderRadius:9,
            }}>
              <span style={{ fontSize:12, color:'var(--text-muted)', fontWeight:500 }}>Dice Score</span>
              <span style={{ fontSize:16, fontWeight:800, color:'var(--neon-green)', fontFamily:'var(--font-mono)' }}>
                {result.dice.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* ══ CLASSIFICATION probabilities ══ */}
      {hasCls && result.tumor_prob != null && (
        <div>
          {hasSeg && <div className="divider" />}
          <div style={{
            fontSize:12, fontWeight:700, color:'var(--text-secondary)',
            marginBottom:10, display:'flex', alignItems:'center', gap:6,
          }}>
            <div style={{ width:10, height:10, borderRadius:2, background:'var(--neon-purple)' }} />
            Classification Output
          </div>
          <ProbChart tumorProb={result.tumor_prob} />
          <div style={{
            display:'flex', justifyContent:'space-between', alignItems:'center',
            marginTop:10, padding:'9px 14px',
            background:'rgba(2,132,199,0.06)',
            border:'1px solid rgba(2,132,199,0.2)',
            borderRadius:9,
          }}>
            <span style={{ fontSize:12, color:'var(--text-muted)', fontWeight:500 }}>Confidence</span>
            <span style={{ fontSize:16, fontWeight:800, color:'var(--electric-blue)', fontFamily:'var(--font-mono)' }}>
              {result.confidence_pct != null
                ? `${Number(result.confidence_pct).toFixed(1)}%`
                : `${(Math.max(result.tumor_prob, 1 - result.tumor_prob) * 100).toFixed(1)}%`}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

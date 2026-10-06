import React from 'react'
import { motion } from 'framer-motion'

export default function ProbChart({ tumorProb }) {
  const absent  = parseFloat(((1 - tumorProb) * 100).toFixed(1))
  const present = parseFloat((tumorProb * 100).toFixed(1))

  const bars = [
    { label: 'Tumor Present', pct: present, color: '#dc2626', bg: 'rgba(220,38,38,0.08)',  border: 'rgba(220,38,38,0.2)'  },
    { label: 'No Tumor',      pct: absent,  color: '#16a34a', bg: 'rgba(22,163,74,0.08)',  border: 'rgba(22,163,74,0.2)'  },
  ]

  return (
    <div className="prob-chart">
      {bars.map(({ label, pct, color, bg, border }) => (
        <div key={label} className="prob-row">
          <div className="prob-label-row">
            <span className="prob-label">{label}</span>
            <span className="prob-pct" style={{ color }}>{pct}%</span>
          </div>
          <div className="prob-bar-bg">
            <motion.div
              className="prob-bar-fill"
              initial={{ width: '0%' }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.9, ease: [0.34, 1.56, 0.64, 1] }}
              style={{ background: color }}
            />
          </div>
        </div>
      ))}

      {/* Verdict badge */}
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        style={{
          marginTop: 12, padding: '10px 14px',
          background: present > 50 ? 'rgba(220,38,38,0.06)' : 'rgba(22,163,74,0.06)',
          border: `1px solid ${present > 50 ? 'rgba(220,38,38,0.2)' : 'rgba(22,163,74,0.2)'}`,
          borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}
      >
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 500 }}>Prediction</span>
        <span style={{
          fontSize: 13, fontWeight: 800,
          color: present > 50 ? '#dc2626' : '#16a34a',
        }}>
          {present > 50 ? '⚠ Tumor Detected' : '✓ No Tumor Detected'}
        </span>
      </motion.div>
    </div>
  )
}

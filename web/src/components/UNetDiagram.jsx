import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

/*
  U-NET DIAGRAM — ALL BLACK
  Every block border, fill, label, arrow and skip connection
  is pure #000000 / near-black. Active state uses bold black.
*/

const LEFT_X = 110
const RIGHT_X = 590
const CX      = 350
const TOP_Y   = 62
const STEP    = 88
const BTK_Y   = TOP_Y + STEP * 4

const NODES = {
  input:      { x: LEFT_X,  y: TOP_Y,          w: 60, h: 48,  label: 'Input',       sub: '1×240×240' },
  enc1:       { x: LEFT_X,  y: TOP_Y+STEP,     w: 60, h: 64,  label: 'Enc 1',       sub: '64 ch'     },
  enc2:       { x: LEFT_X,  y: TOP_Y+STEP*2,   w: 60, h: 78,  label: 'Enc 2',       sub: '128 ch'    },
  enc3:       { x: LEFT_X,  y: TOP_Y+STEP*3,   w: 60, h: 94,  label: 'Enc 3',       sub: '256 ch'    },
  bottleneck: { x: CX,      y: BTK_Y,           w: 74, h: 110, label: 'Bottleneck',  sub: '512 ch'    },
  dec3:       { x: RIGHT_X, y: TOP_Y+STEP*3,   w: 60, h: 94,  label: 'Dec 3',       sub: '256 ch'    },
  dec2:       { x: RIGHT_X, y: TOP_Y+STEP*2,   w: 60, h: 78,  label: 'Dec 2',       sub: '128 ch'    },
  dec1:       { x: RIGHT_X, y: TOP_Y+STEP,     w: 60, h: 64,  label: 'Dec 1',       sub: '64 ch'     },
  output:     { x: RIGHT_X, y: TOP_Y,          w: 60, h: 48,  label: 'Output',      sub: '1×240×240' },
}

const PASS_ORDER = ['input','enc1','enc2','enc3','bottleneck','dec3','dec2','dec1','output']
const SKIP_PAIRS = [
  { enc:'enc1', dec:'dec1', level:1 },
  { enc:'enc2', dec:'dec2', level:2 },
  { enc:'enc3', dec:'dec3', level:3 },
]

/* ── Block ── */
function Block({ id, active, passed }) {
  const n = NODES[id]

  // Active: solid black fill + white text
  // Passed: black border, very light fill, black text
  // Idle:   thin black border, white fill, light gray text
  const fillColor   = active ? '#000000' : passed ? '#1a1a1a' : '#ffffff'
  const fillOpacity = active ? 1 : passed ? 0.06 : 1
  const strokeColor = '#000000'
  const strokeW     = active ? 3 : passed ? 1.8 : 1.2
  const strokeOp    = active ? 1 : passed ? 0.7 : 0.25
  const labelColor  = active ? '#ffffff' : passed ? '#000000' : '#1a1a1a'
  const subColor    = active ? 'rgba(255,255,255,0.85)' : passed ? '#333333' : '#555555'
  const subOp       = active ? 1 : passed ? 0.8 : 0.45

  return (
    <g>
      {/* Pulsing shadow when active */}
      {active && (
        <motion.rect
          x={n.x-n.w/2-6} y={n.y-n.h/2-6}
          width={n.w+12} height={n.h+12} rx={10}
          fill="none" stroke="#000000" strokeWidth={6} opacity={0.08}
          animate={{ opacity:[0.04,0.18,0.04], strokeWidth:[4,8,4] }}
          transition={{ duration:1.1, repeat:Infinity, ease:'easeInOut' }}
          style={{ filter:'blur(4px)' }}
        />
      )}

      {/* Block body */}
      <motion.rect
        x={n.x-n.w/2} y={n.y-n.h/2}
        width={n.w} height={n.h} rx={7}
        fill={fillColor}
        fillOpacity={fillOpacity}
        stroke={strokeColor}
        strokeWidth={strokeW}
        strokeOpacity={strokeOp}
        animate={active ? { strokeOpacity:[0.6,1,0.6] } : {}}
        transition={{ duration:0.9, repeat:active?Infinity:0 }}
        style={{ filter: active ? 'drop-shadow(0 2px 10px rgba(0,0,0,0.35))' : 'none' }}
      />

      {/* Label */}
      <text
        x={n.x} y={n.y - (n.h>60 ? 9 : 4)}
        textAnchor="middle"
        fill={labelColor}
        opacity={active ? 1 : passed ? 0.9 : 0.5}
        fontSize={10.5} fontFamily="JetBrains Mono, monospace" fontWeight="800"
      >
        {n.label}
      </text>
      <text
        x={n.x} y={n.y + (n.h>60 ? 4 : 8)}
        textAnchor="middle"
        fill={subColor}
        opacity={subOp}
        fontSize={8.5} fontFamily="JetBrains Mono, monospace" fontWeight="600"
      >
        {n.sub}
      </text>
      {n.h > 60 && id !== 'input' && id !== 'output' && (
        <text
          x={n.x} y={n.y + (n.h>80 ? 17 : 18)}
          textAnchor="middle"
          fill={active ? 'rgba(255,255,255,0.7)' : '#666666'}
          opacity={active ? 1 : passed ? 0.6 : 0.3}
          fontSize={7} fontFamily="JetBrains Mono, monospace"
        >
          {id === 'bottleneck' ? '2×Conv+BN+ReLU' : 'DoubleConv'}
        </text>
      )}
    </g>
  )
}

/* ── Skip Connection — pure black arc ── */
function SkipConnection({ enc, dec, level, active, ever }) {
  const nE = NODES[enc]
  const nD = NODES[dec]

  const x1 = nE.x + nE.w/2 + 2
  const x2 = nD.x - nD.w/2 - 2
  const y  = nE.y

  // Arc rises above the block, taller for higher levels
  const arcRise = nE.h/2 + 20 + (3 - level) * 18
  const cy = y - arcRise
  const cpX = (x1 + x2) / 2

  const pathD = `M ${x1},${y} C ${x1+60},${cy} ${x2-60},${cy} ${x2},${y}`

  // Active = solid black; ever = medium black; idle = very faint
  const strokeColor = '#000000'
  const strokeOp    = active ? 1 : ever ? 0.4 : 0.12
  const strokeW     = active ? 2.8 : ever ? 1.5 : 1

  return (
    <g>
      {/* Glow blur */}
      {active && (
        <motion.path
          d={pathD} fill="none"
          stroke="#000000" strokeWidth={8} opacity={0.08}
          animate={{ opacity:[0.04,0.18,0.04] }}
          transition={{ duration:1.2, repeat:Infinity }}
          style={{ filter:'blur(5px)' }}
        />
      )}

      {/* Main arc */}
      <motion.path
        d={pathD} fill="none"
        stroke={strokeColor}
        strokeWidth={strokeW}
        strokeOpacity={strokeOp}
        strokeDasharray={active ? '10 5' : '7 5'}
        markerEnd={active ? 'url(#skip-black)' : ever ? 'url(#skip-faded)' : 'url(#skip-dim)'}
        animate={active ? { strokeDashoffset:[0,-30] } : {}}
        transition={{ duration:0.85, repeat:active?Infinity:0, ease:'linear' }}
      />

      {/* Travelling dot */}
      {active && (
        <motion.circle
          r={4} fill="#000000"
          animate={{
            cx:[x1, x1+50, cpX, x2-50, x2],
            cy:[y, cy+18, cy, cy+18, y],
          }}
          transition={{ duration:0.85, repeat:Infinity, ease:'easeInOut' }}
          style={{ filter:'drop-shadow(0 0 3px rgba(0,0,0,0.5))' }}
        />
      )}

      {/* "skip connection" label */}
      <text
        x={cpX} y={cy - 8}
        textAnchor="middle"
        fill="#000000"
        opacity={active ? 0.9 : ever ? 0.45 : 0.15}
        fontSize={7.5} fontFamily="JetBrains Mono, monospace" fontWeight="700"
      >
        skip connection
      </text>

      {/* ⊕ concat symbol at decoder end */}
      <motion.circle
        cx={x2} cy={y} r={9}
        fill={active ? '#000000' : 'transparent'}
        fillOpacity={active ? 0.08 : 0}
        stroke="#000000"
        strokeWidth={active ? 2 : 1}
        strokeOpacity={active ? 1 : ever ? 0.45 : 0.15}
        animate={active ? { r:[8,10,8], strokeOpacity:[0.7,1,0.7] } : {}}
        transition={{ duration:1, repeat:active?Infinity:0 }}
      />
      <text
        x={x2} y={y+5}
        textAnchor="middle"
        fill="#000000"
        opacity={active ? 1 : ever ? 0.45 : 0.15}
        fontSize={11} fontFamily="serif" fontWeight="900"
      >
        ⊕
      </text>

      {/* Source dot on encoder */}
      <circle
        cx={x1} cy={y} r={4}
        fill={active ? '#000000' : 'transparent'}
        stroke="#000000"
        strokeWidth={1.5}
        opacity={active ? 1 : ever ? 0.4 : 0.15}
      />
    </g>
  )
}

/* ── Vertical arrow ── */
function VArrow({ from, to, active }) {
  const n1 = NODES[from], n2 = NODES[to]
  const goDown = n2.y > n1.y
  return (
    <motion.line
      x1={n1.x} y1={goDown ? n1.y+n1.h/2 : n1.y-n1.h/2}
      x2={n2.x} y2={goDown ? n2.y-n2.h/2 : n2.y+n2.h/2}
      stroke="#000000"
      strokeWidth={active ? 2.2 : 0.8}
      strokeOpacity={active ? 1 : 0.2}
      markerEnd={active ? 'url(#arr-black)' : 'url(#arr-dim)'}
    />
  )
}

/* ── Diagonal arrow ── */
function DiagArrow({ from, to, active }) {
  const n1 = NODES[from], n2 = NODES[to]
  return (
    <motion.line
      x1={n1.x+n1.w/2} y1={n1.y+n1.h/2}
      x2={n2.x-n2.w/2} y2={n2.y+n2.h/2}
      stroke="#000000"
      strokeWidth={active ? 2.2 : 0.8}
      strokeOpacity={active ? 1 : 0.2}
      markerEnd={active ? 'url(#arr-black)' : 'url(#arr-dim)'}
    />
  )
}

/* ── Travelling particle ── */
function Particle({ progress }) {
  if (progress <= 0 || progress >= 1) return null
  const total = PASS_ORDER.length - 1
  const si    = Math.min(Math.floor(progress * total), total - 1)
  const t     = progress * total - si
  const n1    = NODES[PASS_ORDER[si]]
  const n2    = NODES[PASS_ORDER[si+1]] || n1
  const x     = n1.x + (n2.x - n1.x) * t
  const y     = n1.y + (n2.y - n1.y) * t
  return (
    <g style={{ filter:'drop-shadow(0 0 5px rgba(0,0,0,0.6))' }}>
      <circle cx={x} cy={y} r={10} fill="#000000" opacity={0.08} />
      <circle cx={x} cy={y} r={5}  fill="#000000" opacity={0.9}  />
      <circle cx={x} cy={y} r={2}  fill="#ffffff" />
    </g>
  )
}

/* ── Side label ── */
function SideLabel({ x, y, text, active, rotate }) {
  return (
    <text x={x} y={y} textAnchor="middle"
      fill="#000000" opacity={active ? 0.6 : 0.15}
      fontSize={7.5} fontFamily="JetBrains Mono, monospace"
      transform={`rotate(${rotate},${x},${y})`}
    >
      {text}
    </text>
  )
}

/* ── MTL CLS Head ── */
function ClsHead({ active }) {
  const bn  = NODES.bottleneck
  const y0  = bn.y + bn.h/2 + 14
  const yBox = y0 + 12
  return (
    <g>
      <motion.line
        x1={bn.x} y1={y0} x2={bn.x} y2={yBox}
        stroke="#000000" strokeWidth={active?2:0.8}
        strokeOpacity={active?1:0.2} strokeDasharray="5 3"
        markerEnd={active?'url(#arr-black)':'none'}
        animate={active?{opacity:[0.6,1,0.6]}:{}}
        transition={{ duration:0.9, repeat:active?Infinity:0 }}
      />
      <motion.rect
        x={bn.x-52} y={yBox} width={104} height={36} rx={7}
        fill="#000000" fillOpacity={active?0.1:0.03}
        stroke="#000000" strokeWidth={active?2:0.8} strokeOpacity={active?1:0.2}
        style={{ filter:active?'drop-shadow(0 2px 8px rgba(0,0,0,0.2))':'none' }}
        animate={active?{strokeOpacity:[0.6,1,0.6]}:{}}
        transition={{ duration:1, repeat:active?Infinity:0 }}
      />
      <text x={bn.x} y={yBox+14} textAnchor="middle"
        fill="#000000" opacity={active?1:0.25}
        fontSize={9.5} fontFamily="JetBrains Mono, monospace" fontWeight="800">
        CLS Head
      </text>
      <text x={bn.x} y={yBox+26} textAnchor="middle"
        fill="#000000" opacity={active?0.65:0.18}
        fontSize={7.5} fontFamily="JetBrains Mono, monospace">
        GAP → Linear → σ
      </text>
    </g>
  )
}

/* ══════════════════════════════════════════════════════════
   Main Export
   ══════════════════════════════════════════════════════════ */
export default function UNetDiagram({ isRunning, variant = 'unet' }) {
  const [activeIdx, setActiveIdx] = useState(-1)
  const [progress,  setProgress]  = useState(0)
  const animRef  = useRef(null)
  const startRef = useRef(null)
  const DURATION = 3800

  useEffect(() => {
    if (!isRunning) {
      setActiveIdx(-1); setProgress(0)
      if (animRef.current) cancelAnimationFrame(animRef.current)
      startRef.current = null; return
    }
    const tick = (ts) => {
      if (!startRef.current) startRef.current = ts
      const p = Math.min((ts - startRef.current) / DURATION, 1)
      setProgress(p)
      setActiveIdx(Math.floor(p * (PASS_ORDER.length - 1)))
      if (p < 1) animRef.current = requestAnimationFrame(tick)
      else setTimeout(() => { setActiveIdx(-1); setProgress(0); startRef.current = null }, 900)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [isRunning])

  const cur  = PASS_ORDER[activeIdx] || null
  const isA  = (id) => id === cur
  const isP  = (id) => PASS_ORDER.indexOf(id) < activeIdx
  const conn = (f, t) => isP(f) && (isA(t) || isP(t))

  const skipActive = (enc) => { const d={enc1:'dec1',enc2:'dec2',enc3:'dec3'}; return isA(d[enc])||isP(d[enc]) }
  const skipEver   = () => isA('bottleneck') || isP('bottleneck')

  const extraH = (variant==='mtl'||variant==='pcgrad') ? 72 : 0
  const viewH  = BTK_Y + NODES.bottleneck.h/2 + 50 + extraH

  return (
    <svg
      viewBox={`-10 30 720 ${viewH}`}
      className="nn-svg"
      style={{ maxHeight: viewH+20, width:'100%', overflow:'visible', background:'transparent' }}
    >
      <defs>
        <marker id="arr-black" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
          <path d="M0,0 L0,8 L8,4 z" fill="#000000" />
        </marker>
        <marker id="arr-dim" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
          <path d="M0,0 L0,7 L7,3.5 z" fill="rgba(0,0,0,0.15)" />
        </marker>
        <marker id="skip-black" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
          <path d="M0,0 L0,8 L8,4 z" fill="#000000" />
        </marker>
        <marker id="skip-faded" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
          <path d="M0,0 L0,7 L7,3.5 z" fill="rgba(0,0,0,0.4)" />
        </marker>
        <marker id="skip-dim" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="rgba(0,0,0,0.12)" />
        </marker>
      </defs>

      {/* U-guide track */}
      <polyline
        points={`${LEFT_X},${TOP_Y-32} ${LEFT_X},${BTK_Y+32} ${CX},${BTK_Y+32} ${RIGHT_X},${BTK_Y+32} ${RIGHT_X},${TOP_Y-32}`}
        fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth={34}
        strokeLinejoin="round" strokeLinecap="round"
      />

      {/* LAYER 1 — Arrows */}
      {['input→enc1','enc1→enc2','enc2→enc3'].map(k => {
        const [f,t]=k.split('→'); return <VArrow key={k} from={f} to={t} active={conn(f,t)} />
      })}
      {['enc1','enc2','enc3'].map((id,i) => (
        <SideLabel key={id}
          x={LEFT_X-42} y={TOP_Y+STEP*(i+0.5)}
          text="MaxPool ↓" rotate={-90}
          active={isP(PASS_ORDER[i])||isA(PASS_ORDER[i+1])}
        />
      ))}
      <DiagArrow from="enc3"       to="bottleneck" active={conn('enc3','bottleneck')} />
      <DiagArrow from="bottleneck" to="dec3"       active={conn('bottleneck','dec3')} />
      {['dec3→dec2','dec2→dec1','dec1→output'].map(k => {
        const [f,t]=k.split('→'); return <VArrow key={k} from={f} to={t} active={conn(f,t)} />
      })}
      {['dec3','dec2','dec1'].map((id,i) => (
        <SideLabel key={id}
          x={RIGHT_X+42} y={TOP_Y+STEP*(3-i-0.5)}
          text="UpSample ↑" rotate={90}
          active={isA(id)||isP(id)}
        />
      ))}

      {/* LAYER 2 — Blocks */}
      {PASS_ORDER.map(id => (
        <Block key={id} id={id} active={isA(id)} passed={isP(id)} />
      ))}

      {/* LAYER 3 — Skip connections (on top) */}
      {SKIP_PAIRS.map(({ enc, dec, level }) => (
        <SkipConnection
          key={enc} enc={enc} dec={dec} level={level}
          active={skipActive(enc)} ever={skipEver()}
        />
      ))}

      {/* LAYER 4 — MTL head */}
      {(variant==='mtl'||variant==='pcgrad') && (
        <ClsHead active={isA('bottleneck')||isP('bottleneck')} />
      )}

      {/* LAYER 5 — Particle */}
      <Particle progress={progress} />
    </svg>
  )
}

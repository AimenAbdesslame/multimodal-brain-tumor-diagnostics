import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

/*
  HALF-U CLASSIFICATION DIAGRAM — ALL BLACK
  Encoder arm descends on left, CLS pipeline runs horizontally.
  Every element is pure #000000 on white.
*/

const LEFT_X = 130
const TOP_Y  = 55
const STEP   = 88
const BW     = 62

const BTK_X = LEFT_X + 180
const BTK_Y = TOP_Y + STEP * 3

const NODES = {
  input:      { x: LEFT_X,        y: TOP_Y,        w: BW, h: 48  },
  enc1:       { x: LEFT_X,        y: TOP_Y+STEP,   w: BW, h: 64  },
  enc2:       { x: LEFT_X,        y: TOP_Y+STEP*2, w: BW, h: 78  },
  enc3:       { x: LEFT_X,        y: BTK_Y,         w: BW, h: 94  },
  bottleneck: { x: BTK_X,         y: BTK_Y,         w: 68, h: 94  },
  gap:        { x: BTK_X+140,     y: BTK_Y,         w: 60, h: 50  },
  mlp:        { x: BTK_X+260,     y: BTK_Y,         w: 60, h: 50  },
  output:     { x: BTK_X+380,     y: BTK_Y,         w: 60, h: 50  },
}

const LABELS = {
  input:      { top:'Input',      bot:'1×240×240' },
  enc1:       { top:'Enc 1',      bot:'64 ch'     },
  enc2:       { top:'Enc 2',      bot:'128 ch'    },
  enc3:       { top:'Enc 3',      bot:'256 ch'    },
  bottleneck: { top:'Bottleneck', bot:'512 ch'    },
  gap:        { top:'GAP',        bot:'512→1'     },
  mlp:        { top:'MLP',        bot:'128→1'     },
  output:     { top:'Logit',      bot:'σ → p'     },
}

const PASS_ORDER  = ['input','enc1','enc2','enc3','bottleneck','gap','mlp','output']
const FROZEN_IDS  = ['enc1','enc2','enc3','bottleneck']

function Block({ id, active, passed, frozen }) {
  const n   = NODES[id]
  const lbl = LABELS[id]

  const fillColor = active ? '#000000' : '#ffffff'
  const fillOp    = active ? 1 : passed ? 0.05 : 1
  const strokeW   = active ? 3 : passed ? 1.8 : 1.2
  const strokeOp  = active ? 1 : passed ? 0.7 : 0.25
  const labelCol  = active ? '#ffffff' : '#000000'
  const labelOp   = active ? 1 : passed ? 0.9 : 0.55
  const subOp     = active ? 0.85 : passed ? 0.7 : 0.4

  return (
    <g>
      {active && (
        <motion.rect
          x={n.x-n.w/2-7} y={n.y-n.h/2-7}
          width={n.w+14} height={n.h+14} rx={11}
          fill="none" stroke="#000000" strokeWidth={6} opacity={0.07}
          animate={{ opacity:[0.03,0.16,0.03] }}
          transition={{ duration:1.1, repeat:Infinity }}
          style={{ filter:'blur(5px)' }}
        />
      )}

      <motion.rect
        x={n.x-n.w/2} y={n.y-n.h/2}
        width={n.w} height={n.h} rx={7}
        fill={fillColor} fillOpacity={fillOp}
        stroke="#000000" strokeWidth={strokeW} strokeOpacity={strokeOp}
        animate={active?{strokeOpacity:[0.6,1,0.6]}:{}}
        transition={{ duration:0.9, repeat:active?Infinity:0 }}
        style={{ filter:active?'drop-shadow(0 2px 10px rgba(0,0,0,0.3))':'none' }}
      />

      {/* Frozen dashed overlay */}
      {frozen && (
        <rect
          x={n.x-n.w/2+3} y={n.y-n.h/2+3}
          width={n.w-6} height={n.h-6} rx={5}
          fill="none" stroke="#000000"
          strokeWidth={1} strokeDasharray="3 2"
          opacity={active ? 0.5 : passed ? 0.3 : 0.15}
        />
      )}

      <text x={n.x} y={n.y-(n.h>55?9:3)} textAnchor="middle"
        fill={labelCol} opacity={labelOp}
        fontSize={10.5} fontFamily="JetBrains Mono, monospace" fontWeight="800">
        {lbl.top}
      </text>
      <text x={n.x} y={n.y+(n.h>55?4:9)} textAnchor="middle"
        fill={active?'rgba(255,255,255,0.85)':'#000000'} opacity={subOp}
        fontSize={8.5} fontFamily="JetBrains Mono, monospace" fontWeight="600">
        {lbl.bot}
      </text>
      {n.h > 60 && id !== 'input' && id !== 'gap' && id !== 'mlp' && id !== 'output' && (
        <text x={n.x} y={n.y+(n.h>80?17:18)} textAnchor="middle"
          fill={active?'rgba(255,255,255,0.65)':'#555555'}
          opacity={active?1:passed?0.55:0.3}
          fontSize={7} fontFamily="JetBrains Mono, monospace">
          {id==='bottleneck'?'2×Conv+BN+ReLU':'DoubleConv'}
        </text>
      )}
      {frozen && (active||passed) && (
        <text x={n.x} y={n.y+n.h/2+12} textAnchor="middle"
          fill="#000000" opacity={0.55}
          fontSize={7} fontFamily="JetBrains Mono, monospace">
          🔒 frozen
        </text>
      )}
    </g>
  )
}

function Particle({ progress }) {
  if (progress<=0||progress>=1) return null
  const total=PASS_ORDER.length-1
  const si=Math.min(Math.floor(progress*total),total-1)
  const t=progress*total-si
  const n1=NODES[PASS_ORDER[si]]; const n2=NODES[PASS_ORDER[si+1]]||n1
  const x=n1.x+(n2.x-n1.x)*t; const y=n1.y+(n2.y-n1.y)*t
  return (
    <g style={{ filter:'drop-shadow(0 0 4px rgba(0,0,0,0.5))' }}>
      <circle cx={x} cy={y} r={9} fill="#000000" opacity={0.08} />
      <circle cx={x} cy={y} r={5} fill="#000000" opacity={0.9}  />
      <circle cx={x} cy={y} r={2} fill="#ffffff" />
    </g>
  )
}

function Arrow({ x1,y1,x2,y2,active }) {
  return (
    <motion.line x1={x1} y1={y1} x2={x2} y2={y2}
      stroke="#000000"
      strokeWidth={active?2.2:0.8}
      strokeOpacity={active?1:0.2}
      markerEnd={active?'url(#cls-arr-on)':'url(#cls-arr-off)'}
    />
  )
}

export default function ClassificationDiagram({ isRunning, frozen=false }) {
  const [activeIdx,setActiveIdx] = useState(-1)
  const [progress, setProgress]  = useState(0)
  const animRef=useRef(null); const startRef=useRef(null)
  const DURATION=3200

  useEffect(() => {
    if (!isRunning) {
      setActiveIdx(-1); setProgress(0)
      if (animRef.current) cancelAnimationFrame(animRef.current)
      startRef.current=null; return
    }
    const tick=(ts)=>{
      if (!startRef.current) startRef.current=ts
      const p=Math.min((ts-startRef.current)/DURATION,1)
      setProgress(p)
      setActiveIdx(Math.floor(p*(PASS_ORDER.length-1)))
      if (p<1) animRef.current=requestAnimationFrame(tick)
      else setTimeout(()=>{setActiveIdx(-1);setProgress(0);startRef.current=null},800)
    }
    animRef.current=requestAnimationFrame(tick)
    return ()=>{ if (animRef.current) cancelAnimationFrame(animRef.current) }
  },[isRunning])

  const cur=PASS_ORDER[activeIdx]||null
  const isA=(id)=>id===cur
  const isP=(id)=>PASS_ORDER.indexOf(id)<activeIdx
  const conn=(f,t)=>isP(f)&&(isA(t)||isP(t))

  const viewW=BTK_X+380+64; const viewH=BTK_Y+NODES.bottleneck.h/2+55

  return (
    <svg viewBox={`-15 20 ${viewW} ${viewH}`} className="nn-svg"
      style={{ maxHeight:viewH+10, width:'100%', overflow:'visible', background:'transparent' }}>
      <defs>
        <marker id="cls-arr-on" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
          <path d="M0,0 L0,8 L8,4 z" fill="#000000" />
        </marker>
        <marker id="cls-arr-off" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="rgba(0,0,0,0.15)" />
        </marker>
      </defs>

      {/* U-arm track */}
      <polyline
        points={`${LEFT_X},${TOP_Y-28} ${LEFT_X},${BTK_Y+30} ${viewW-20},${BTK_Y+30}`}
        fill="none" stroke="rgba(0,0,0,0.05)" strokeWidth={30}
        strokeLinejoin="round" strokeLinecap="round"
      />

      {/* MaxPool labels */}
      {['enc1','enc2','enc3'].map((id,i)=>(
        <text key={id}
          x={LEFT_X-38} y={TOP_Y+STEP*(i+0.5)}
          textAnchor="middle" fill="#000000"
          opacity={isP(PASS_ORDER[i])||isA(PASS_ORDER[i+1])?0.55:0.15}
          fontSize={7.5} fontFamily="JetBrains Mono, monospace"
          transform={`rotate(-90,${LEFT_X-38},${TOP_Y+STEP*(i+0.5)})`}>
          MaxPool ↓
        </text>
      ))}

      {/* Encoder DOWN arrows */}
      {['input→enc1','enc1→enc2','enc2→enc3'].map(k=>{
        const [f,t]=k.split('→'); const n1=NODES[f],n2=NODES[t]
        return <Arrow key={k} x1={n1.x} y1={n1.y+n1.h/2} x2={n2.x} y2={n2.y-n2.h/2} active={conn(f,t)} />
      })}

      {/* enc3 → bottleneck */}
      {(()=>{const f='enc3',t='bottleneck',n1=NODES[f],n2=NODES[t];
        return <Arrow x1={n1.x+n1.w/2} y1={n1.y} x2={n2.x-n2.w/2} y2={n2.y} active={conn(f,t)} />
      })()}

      {/* Hook label */}
      {(isA('bottleneck')||isP('bottleneck'))&&(
        <g>
          <line x1={BTK_X} y1={NODES.bottleneck.y-NODES.bottleneck.h/2}
            x2={BTK_X} y2={NODES.bottleneck.y-NODES.bottleneck.h/2-28}
            stroke="#000000" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.7}/>
          <text x={BTK_X} y={NODES.bottleneck.y-NODES.bottleneck.h/2-32}
            textAnchor="middle" fill="#000000" opacity={0.7}
            fontSize={8} fontFamily="JetBrains Mono, monospace" fontWeight="700">
            forward hook
          </text>
        </g>
      )}

      {/* Horizontal CLS pipeline arrows */}
      {['bottleneck→gap','gap→mlp','mlp→output'].map(k=>{
        const [f,t]=k.split('→'); const n1=NODES[f],n2=NODES[t]
        return <Arrow key={k} x1={n1.x+n1.w/2} y1={n1.y} x2={n2.x-n2.w/2} y2={n2.y} active={conn(f,t)} />
      })}

      {/* Blocks */}
      {PASS_ORDER.map(id=>(
        <Block key={id} id={id} active={isA(id)} passed={isP(id)}
          frozen={frozen&&FROZEN_IDS.includes(id)} />
      ))}

      <Particle progress={progress} />
    </svg>
  )
}

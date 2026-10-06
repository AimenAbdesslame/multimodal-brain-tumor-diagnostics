import React, { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'

/*
  =================================================================
  SEGMENTATION VISUALIZATION — mirrors notebook cell 6 exactly

  Notebook code (per sample):
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(gt,   cmap="Greens", alpha=0.5*(gt>0),   vmin=0,vmax=1)
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred, cmap="Reds",   alpha=0.5*(pred>0), vmin=0,vmax=1)
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(gt,   cmap="Greens", alpha=0.5*(gt>0),   vmin=0,vmax=1)
    axes[3].imshow(pred, cmap="Reds",   alpha=0.5*(pred>0), vmin=0,vmax=1)

  Canvas translation:
    1. Draw MRI image scaled to S×S with grayscale filter
    2. Read pixel data from canvas
    3. For each pixel where mask[i] > 0.5:
         blend pixel color with overlay color at alpha=0.5
         (same as matplotlib alpha=0.5*(mask>0))
    4. putImageData back

  Colors (approximating matplotlib colormaps on binary masks):
    cmap="Greens" vmax=1 → value 1 → dark green  RGB(0, 150, 60)
    cmap="Reds"   vmax=1 → value 1 → dark red     RGB(200, 30, 30)
  =================================================================
*/

const S = 200  // canvas panel size in px (each of the 4 panels)

/* ── Blend a color onto grayscale canvas pixels where mask==1 ── */
function blendMaskOnPixels(ctx, dx, dy, size, maskArr, R, G, B, alpha = 0.5) {
  const id = ctx.getImageData(dx, dy, size, size)
  const d  = id.data
  for (let i = 0; i < maskArr.length; i++) {
    if (maskArr[i] > 0.5) {
      d[i*4]     = Math.round(d[i*4]     * (1 - alpha) + R * alpha)
      d[i*4 + 1] = Math.round(d[i*4 + 1] * (1 - alpha) + G * alpha)
      d[i*4 + 2] = Math.round(d[i*4 + 2] * (1 - alpha) + B * alpha)
      // d[i*4+3] stays 255
    }
  }
  ctx.putImageData(id, dx, dy)
}

/* ── Draw MRI image as grayscale onto (dx,dy) panel ── */
function drawGrayPanel(ctx, img, dx, dy, size) {
  ctx.save()
  ctx.filter = 'grayscale(1) brightness(1.05)'
  ctx.drawImage(img, dx, dy, size, size)
  ctx.filter = 'none'
  ctx.restore()
}

/* ── Panel header (title bar at top) ── */
function drawHeader(ctx, text, dx, dy, size, sub = null) {
  const BAR = 26
  ctx.save()
  ctx.fillStyle = 'rgba(15,23,42,0.82)'
  ctx.fillRect(dx, dy, size, BAR)
  ctx.fillStyle = '#ffffff'
  ctx.font = `bold 10px "JetBrains Mono", monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, dx + size/2, dy + (sub ? 9 : BAR/2))
  if (sub) {
    ctx.font = `9px "JetBrains Mono", monospace`
    ctx.fillStyle = 'rgba(255,255,255,0.75)'
    ctx.fillText(sub, dx + size/2, dy + 19)
  }
  ctx.restore()
}

/* ── Thin separator line between panels ── */
function drawSep(ctx, x, dy, size) {
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.5)'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(x, dy)
  ctx.lineTo(x, dy + size)
  ctx.stroke()
  ctx.restore()
}

/* ── Generate a synthetic ellipse mask (demo prediction) ── */
function makePredMask(size) {
  const mask = new Float32Array(size * size)
  const cx = size * 0.53, cy = size * 0.47
  const rx = size * 0.12, ry = size * 0.095
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const ndx = (x - cx) / rx, ndy = (y - cy) / ry
      if (ndx*ndx + ndy*ndy <= 1.0) mask[y*size + x] = 1
    }
  }
  return mask
}

/* ── Generate a synthetic GT mask (slightly larger ellipse) ── */
function makeGTMask(size) {
  const mask = new Float32Array(size * size)
  const cx = size * 0.52, cy = size * 0.46
  const rx = size * 0.155, ry = size * 0.125
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const ndx = (x - cx) / rx, ndy = (y - cy) / ry
      if (ndx*ndx + ndy*ndy <= 1.0) mask[y*size + x] = 1
    }
  }
  return mask
}

/* ── Decode a base64 PNG mask → Float32Array binary mask ── */
async function decodeMaskB64(b64, size) {
  return new Promise(resolve => {
    const img = new window.Image()
    img.onload = () => {
      const oc   = document.createElement('canvas')
      oc.width   = oc.height = size
      const octx = oc.getContext('2d')
      octx.drawImage(img, 0, 0, size, size)
      const px   = octx.getImageData(0, 0, size, size).data
      const mask = new Float32Array(size * size)
      // backend encodes mask as white on black, or uses red channel > 0
      for (let i = 0; i < size*size; i++) {
        mask[i] = px[i*4] > 127 ? 1 : 0   // red channel threshold
      }
      resolve(mask)
    }
    img.onerror  = () => resolve(null)
    img.src      = `data:image/png;base64,${b64}`
  })
}

/* ── Decode a File/Blob GT mask ── */
async function decodeGTFile(file, size) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file)
    const img = new window.Image()
    img.onload = () => {
      URL.revokeObjectURL(url)
      const oc   = document.createElement('canvas')
      oc.width   = oc.height = size
      const octx = oc.getContext('2d')
      // convert to grayscale then threshold
      octx.filter = 'grayscale(1)'
      octx.drawImage(img, 0, 0, size, size)
      octx.filter = 'none'
      const px   = octx.getImageData(0, 0, size, size).data
      const mask = new Float32Array(size * size)
      for (let i = 0; i < size*size; i++) {
        mask[i] = px[i*4] > 127 ? 1 : 0
      }
      resolve(mask)
    }
    img.onerror = () => { URL.revokeObjectURL(url); resolve(null) }
    img.src = url
  })
}

/*
  =================================================================
  Main component: SegmentationViz
  Props:
    imageUrl   — blob URL of uploaded MRI image
    diceScore  — float dice score (optional)
    maskB64    — base64-encoded predicted mask PNG from API (optional)
    gtFile     — File object of the uploaded ground truth mask (optional)
  =================================================================
*/
export function SegmentationViz({ imageUrl, diceScore, maskB64 = null, gtFile = null }) {
  const canvasRef = useRef(null)
  const [status,  setStatus]  = useState('idle')  // idle | loading | done | error

  useEffect(() => {
    if (!imageUrl) { setStatus('idle'); return }
    setStatus('loading')

    const mri = new window.Image()
    mri.crossOrigin = 'anonymous'
    mri.src = imageUrl

    mri.onload = async () => {
      try {
        // ── Decode masks ──────────────────────────────────
        let predMask = null
        if (maskB64) predMask = await decodeMaskB64(maskB64, S)
        if (!predMask) predMask = makePredMask(S)  // demo ellipse

        let gtMask = null
        if (gtFile) gtMask = await decodeGTFile(gtFile, S)
        if (!gtMask) gtMask = makeGTMask(S)  // demo GT ellipse (slightly bigger)

        // ── Setup canvas (4 panels wide) ─────────────────
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.width  = S * 4
        canvas.height = S
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // ══ PANEL 0: Input MRI — axes[0].imshow(image, cmap="gray") ══
        drawGrayPanel(ctx, mri, 0, 0, S)
        drawHeader(ctx, 'Input MRI', 0, 0, S)
        drawSep(ctx, S, 0, S)

        // ══ PANEL 1: Ground Truth ══════════════════════════════
        // axes[1].imshow(image, cmap="gray")
        // axes[1].imshow(gt, cmap="Greens", alpha=0.5*(gt>0), vmin=0,vmax=1)
        drawGrayPanel(ctx, mri, S, 0, S)
        // cmap="Greens" value=1 → dark green R=0,G=150,B=60
        blendMaskOnPixels(ctx, S, 0, S, gtMask, 0, 150, 60, 0.5)
        drawHeader(ctx, 'Ground Truth Mask', S, 0, S,
          gtFile ? null : '(demo)')
        drawSep(ctx, S*2, 0, S)

        // ══ PANEL 2: U-Net Prediction ═══════════════════════════
        // axes[2].imshow(image, cmap="gray")
        // axes[2].imshow(pred, cmap="Reds", alpha=0.5*(pred>0), vmin=0,vmax=1)
        drawGrayPanel(ctx, mri, S*2, 0, S)
        // cmap="Reds" value=1 → dark red R=200,G=30,B=30
        blendMaskOnPixels(ctx, S*2, 0, S, predMask, 200, 30, 30, 0.5)
        const diceLabel = diceScore != null
          ? `Dice: ${diceScore.toFixed(4)}`
          : null
        drawHeader(ctx, 'U-Net Prediction', S*2, 0, S, diceLabel)
        drawSep(ctx, S*3, 0, S)

        // ══ PANEL 3: Combined Overlay ════════════════════════════
        // axes[3].imshow(image, cmap="gray")
        // axes[3].imshow(gt,   cmap="Greens", alpha=0.5*(gt>0),   ...)
        // axes[3].imshow(pred, cmap="Reds",   alpha=0.5*(pred>0), ...)
        drawGrayPanel(ctx, mri, S*3, 0, S)
        blendMaskOnPixels(ctx, S*3, 0, S, gtMask,   0,   150, 60,  0.45)
        blendMaskOnPixels(ctx, S*3, 0, S, predMask, 200, 30,  30,  0.45)
        drawHeader(ctx, 'Combined Overlay', S*3, 0, S)

        setStatus('done')
      } catch (e) {
        console.error('[SegmentationViz]', e)
        setStatus('error')
      }
    }
    mri.onerror = () => setStatus('error')
  }, [imageUrl, maskB64, gtFile, diceScore])

  if (!imageUrl) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ width: '100%' }}
    >
      {/* Legend — matches notebook color scheme */}
      <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:8, flexWrap:'wrap' }}>
        <span style={{ fontSize:11, fontWeight:700, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.06em' }}>
          Legend:
        </span>
        <div style={{ display:'flex', alignItems:'center', gap:5 }}>
          <div style={{ width:14, height:14, borderRadius:3, background:'rgb(0,150,60)', opacity:0.85 }} />
          <span style={{ fontSize:11, color:'var(--text-secondary)', fontWeight:600 }}>Ground Truth (Greens)</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:5 }}>
          <div style={{ width:14, height:14, borderRadius:3, background:'rgb(200,30,30)', opacity:0.85 }} />
          <span style={{ fontSize:11, color:'var(--text-secondary)', fontWeight:600 }}>Prediction (Reds)</span>
        </div>
        {diceScore != null && (
          <span style={{
            marginLeft:'auto',
            fontFamily:'var(--font-mono)', fontWeight:800, fontSize:13,
            color:'var(--neon-green)',
            padding:'2px 12px',
            background:'rgba(22,163,74,0.08)',
            border:'1px solid rgba(22,163,74,0.22)',
            borderRadius:100,
          }}>
            Dice: {diceScore.toFixed(4)}
          </span>
        )}
      </div>

      {/* Canvas */}
      <div style={{
        width:'100%', borderRadius:10, overflow:'hidden',
        border:'1px solid var(--border-subtle)',
        background:'#0f172a',
        boxShadow:'0 2px 12px rgba(0,0,0,0.1)',
        position:'relative',
      }}>
        {status === 'loading' && (
          <div style={{
            position:'absolute', inset:0,
            display:'flex', alignItems:'center', justifyContent:'center',
            background:'rgba(15,23,42,0.6)', color:'#fff',
            fontSize:12, fontFamily:'var(--font-mono)',
          }}>
            Rendering segmentation panels…
          </div>
        )}
        <canvas
          ref={canvasRef}
          style={{ display:'block', width:'100%', height:'auto' }}
        />
      </div>

      {/* Column sub-labels */}
      {status === 'done' && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', marginTop:5, gap:2 }}>
          {[
            'Input MRI',
            `Ground Truth${gtFile ? '' : ' (demo)'}`,
            `Prediction${diceScore!=null?' (Dice: '+diceScore.toFixed(3)+')':''}`,
            'Combined Overlay',
          ].map(l => (
            <div key={l} style={{
              textAlign:'center', fontSize:9.5,
              color:'var(--text-muted)',
              fontFamily:'var(--font-mono)',
              lineHeight:1.4,
            }}>
              {l}
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

/* Back-compat aliases */
export function DemoMaskOverlay({ imageUrl, diceScore, maskB64 }) {
  return <SegmentationViz imageUrl={imageUrl} diceScore={diceScore} maskB64={maskB64} />
}
export function Base64MaskOverlay({ imageUrl, base64Mask, diceScore }) {
  return <SegmentationViz imageUrl={imageUrl} diceScore={diceScore} maskB64={base64Mask} />
}

import React, { useState, useRef } from 'react'
import { Upload, X, Image as ImageIcon } from 'lucide-react'

export default function Dropzone({ onFile, accept = '.png,.jpg,.jpeg,.npz', label = 'MRI Slice' }) {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    onFile(f)
    if (f.type.startsWith('image/')) {
      const url = URL.createObjectURL(f)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }

  const clear = (e) => {
    e.stopPropagation()
    setFile(null)
    setPreview(null)
    onFile(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div
      className={`dropzone${isDragging ? ' drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {file ? (
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          {preview ? (
            <img src={preview} alt="preview" style={{ maxHeight: 100, borderRadius: 6, objectFit: 'contain' }} />
          ) : (
            <div style={{ color: 'var(--electric-blue)', opacity: 0.7 }}>
              <ImageIcon size={32} />
            </div>
          )}
          <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {file.name}
          </span>
          <button
            onClick={clear}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, padding: '4px 10px',
              background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.25)',
              borderRadius: 6, color: 'var(--neon-red)', fontSize: 11, cursor: 'pointer'
            }}
          >
            <X size={11} /> Remove
          </button>
        </div>
      ) : (
        <>
          <Upload size={28} className="dropzone-icon" />
          <p>Drop your <span>{label}</span> here</p>
          <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>or <span>click to browse</span></p>
          <span className="dropzone-formats">Supports .png .jpg .npz</span>
        </>
      )}
    </div>
  )
}

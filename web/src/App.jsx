import React, { useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain, BarChart2, Layers, Zap, Activity, Trophy, Radio
} from 'lucide-react'

import UNetPage from './pages/UNetPage'
import ClassificationPage from './pages/ClassificationPage'
import TransferLearningPage from './pages/TransferLearningPage'
import StandardMTLPage from './pages/StandardMTLPage'
import PCGradMTLPage from './pages/PCGradMTLPage'
import BenchmarkPage from './pages/BenchmarkPage'

const NAV_ITEMS = [
  { path: '/', label: 'UNet Segmentation', icon: Brain, badge: 'SEG' },
  { path: '/classification', label: 'Classification', icon: BarChart2, badge: 'CLS' },
  { path: '/transfer', label: 'Transfer Learning', icon: Layers, badge: 'TL' },
  { path: '/mtl', label: 'Standard MTL', icon: Activity, badge: 'MTL' },
  { path: '/pcgrad', label: 'PCGrad MTL', icon: Zap, badge: 'NEW', badgeClass: 'green' },
  { path: '/benchmark', label: 'Benchmark Suite', icon: Trophy, badge: 'LIVE', badgeClass: 'green' },
]

function Sidebar({ backendStatus }) {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <NavLink to="/" className="logo-mark">
          <div className="logo-icon">
            <Radio size={18} color="#fff" />
          </div>
          <div className="logo-text">
            <span className="name">NeuroScan AI</span>
            <span className="tagline">Brain Tumor Diagnostics</span>
          </div>
        </NavLink>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Models</div>
        {NAV_ITEMS.map(({ path, label, icon: Icon, badge, badgeClass }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon size={15} className="nav-icon" />
            {label}
            {badge && (
              <span className={`nav-badge${badgeClass ? ' ' + badgeClass : ''}`}>
                {badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-pill">
          <span className={`status-dot${backendStatus === 'online' ? '' : ' offline'}`} />
          <span>API {backendStatus === 'online' ? 'Online' : 'Offline — Demo Mode'}</span>
        </div>
      </div>
    </aside>
  )
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const location = useLocation()

  useEffect(() => {
    fetch('/api/metrics')
      .then(r => r.ok ? setBackendStatus('online') : setBackendStatus('offline'))
      .catch(() => setBackendStatus('offline'))
  }, [])

  return (
    <div className="app-layout">
      <Sidebar backendStatus={backendStatus} />
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
          >
            <Routes location={location}>
              <Route path="/" element={<UNetPage backendStatus={backendStatus} />} />
              <Route path="/classification" element={<ClassificationPage backendStatus={backendStatus} />} />
              <Route path="/transfer" element={<TransferLearningPage backendStatus={backendStatus} />} />
              <Route path="/mtl" element={<StandardMTLPage backendStatus={backendStatus} />} />
              <Route path="/pcgrad" element={<PCGradMTLPage backendStatus={backendStatus} />} />
              <Route path="/benchmark" element={<BenchmarkPage backendStatus={backendStatus} />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}

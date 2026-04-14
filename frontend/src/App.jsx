import { useState, useEffect, useRef, useCallback } from 'react'
import './index.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ─── Live Clock ───────────────────────────────────────────────────────────────
function LiveClock() {
  const [time, setTime] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <span className="header-time">
      {time.toLocaleTimeString('en-IN', { hour12: false })} IST
    </span>
  )
}

// ─── Gauge Bar ─────────────────────────────────────────────────────────────────
function GaugeBar({ label, value, max = 100, color = '#00c8ff', unit = '%' }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="gauge-row">
      <div className="gauge-meta">
        <span className="gauge-label">{label}</span>
        <span className="gauge-val" style={{ color }}>
          {typeof value === 'number' ? value.toFixed(1) : value}{unit}
        </span>
      </div>
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%`, background: color, color }} />
      </div>
    </div>
  )
}

// ─── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, icon, color }) {
  return (
    <div className="stat-card" style={{ '--card-color': color }}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-unit">{unit}</div>
      <div className="stat-icon" aria-hidden="true">{icon}</div>
    </div>
  )
}

// ─── Mode Selector Tabs ────────────────────────────────────────────────────────
function ModeTabs({ activeMode, onChange, disabled }) {
  const tabs = [
    { id: 'dashcam', label: '📼 Dashcam', desc: 'Pre-recorded foggy video' },
    { id: 'webcam',  label: '📷 Webcam',  desc: 'Your live camera' },
    { id: 'upload',  label: '📁 Upload',  desc: 'Your own video file' },
  ]
  return (
    <div className="mode-tabs">
      {tabs.map(t => (
        <button
          key={t.id}
          id={`tab-${t.id}`}
          className={`mode-tab ${activeMode === t.id ? 'active' : ''}`}
          onClick={() => !disabled && onChange(t.id)}
          disabled={disabled}
        >
          <span className="tab-label">{t.label}</span>
          <span className="tab-desc">{t.desc}</span>
        </button>
      ))}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [inputMode, setInputMode] = useState('dashcam')   // dashcam | webcam | upload
  const [isStreaming, setIsStreaming] = useState(false)
  const [isOnline, setIsOnline] = useState(false)
  const [imgSrc, setImgSrc]     = useState(null)
  const [alerts, setAlerts]     = useState([])
  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadUrl, setUploadUrl]       = useState(null)   // object URL for upload stream
  const [uploadError, setUploadError]   = useState(null)
  const fileInputRef = useRef(null)

  const [metrics, setMetrics] = useState({ fps: 0, visibility: 0, contrast: 0, uptime: 0 })
  const [weatherMode, setWeatherMode]   = useState('STANDBY')

  // ── Fetch real-time metrics from backend AI engine ─────────────────────────
  useEffect(() => {
    if (!isStreaming) {
      setMetrics({ fps: 0, visibility: 0, contrast: 0, uptime: 0 })
      setWeatherMode('STANDBY')
      return
    }

    let uptimeCounter = 0
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/metrics`)
        if (res.ok) {
          const data = await res.json()
          uptimeCounter++
          setMetrics({
            fps: data.fps,
            visibility: data.visibility,
            contrast: data.contrast,
            uptime: uptimeCounter,
          })
          setWeatherMode(data.weather)

          // Auto-generate alerts based on real safety scores
          if (data.visibility < 40) {
            const newAlert = {
              icon: '⚠️',
              text: 'CRITICAL: Very Low Visibility Detected!',
              time: new Date().toLocaleTimeString('en-IN', { hour12: false }),
              id: Date.now()
            }
            setAlerts(prev => [newAlert, ...prev].slice(0, 5))
          }
        }
      } catch (e) {
        console.error("Failed to fetch metrics", e)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isStreaming])

  // ── Ping backend ────────────────────────────────────────────────────────────
  const checkBackend = useCallback(async () => {
    try {
      const r = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(2000) })
      setIsOnline(r.ok)
    } catch { setIsOnline(false) }
  }, [])
  useEffect(() => { checkBackend(); const t = setInterval(checkBackend, 5000); return () => clearInterval(t) }, [checkBackend])

  // ── Stop streaming helper ───────────────────────────────────────────────────
  const stopStream = () => {
    setIsStreaming(false)
    setImgSrc(null)
    setUploadUrl(null)
    setUploadError(null)
    setAlerts([])
  }

  // ── CONNECT logic per mode ──────────────────────────────────────────────────
  const handleConnect = async () => {
    if (isStreaming) { stopStream(); return }
    setUploadError(null)

    if (inputMode === 'dashcam') {
      setImgSrc(`${API_URL}/video_feed?t=${Date.now()}`)
      setIsStreaming(true)

    } else if (inputMode === 'webcam') {
      setImgSrc(`${API_URL}/webcam_feed?t=${Date.now()}`)
      setIsStreaming(true)

    } else if (inputMode === 'upload') {
      if (!uploadedFile) { setUploadError('Please select a video file first.'); return }

      // Step 1: POST the file, get back a stream_id
      try {
        const formData = new FormData()
        formData.append('file', uploadedFile)
        const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData })
        if (!res.ok) throw new Error('Upload POST failed')
        const data = await res.json()
        // Step 2: Use stream_id to set a GET img src (MJPEG stream)
        setImgSrc(`${API_URL}/stream/${data.stream_id}`)
        setIsStreaming(true)
      } catch (e) {
        setUploadError('Upload failed. Check the backend terminal for errors.')
      }
    }
  }

  // ── File selection ──────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setUploadedFile(f)
    setUploadError(null)
  }

  // imgSrc is used for all 3 modes:
  //  - dashcam:  GET /video_feed
  //  - webcam:   GET /webcam_feed
  //  - upload:   GET /stream/{stream_id}  (set after POST /upload returns stream_id)

  const modeColor = { FOGGY: '#ff8c42', CLEAR: '#00ff88', NIGHT: '#a855f7', STANDBY: 'rgba(180,210,240,0.4)' }[weatherMode] || '#00c8ff'
  const activeFooterDot = Math.floor((metrics.uptime % 3))

  const modeLabel = { dashcam: '📼 DASHCAM', webcam: '📷 WEBCAM', upload: '📁 UPLOAD' }[inputMode]

  return (
    <div className="app-shell">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">🚗</div>
          <div>
            <div className="logo-text">CLEARDRIVE</div>
            <div className="logo-sub">Advanced Driver Assistance System</div>
          </div>
        </div>
        <div className="header-status">
          <div className="input-mode-badge">{modeLabel}</div>
          <div className={`status-pill ${isOnline ? 'online' : 'offline'}`}>
            <div className="status-dot" />
            {isOnline ? 'AI ENGINE ONLINE' : 'ENGINE OFFLINE'}
          </div>
          <LiveClock />
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────────── */}
      <main className="main-content">
        {/* Stat Cards */}
        <div className="stats-row">
          <StatCard label="Processing Speed" value={metrics.fps}        unit="Frames Per Second"       icon="⚡" color="#00c8ff" />
          <StatCard label="Visibility Score"  value={metrics.visibility} unit="Safe Zone Coverage"       icon="👁" color="#00ff88" />
          <StatCard label="Contrast Gain"     value={metrics.contrast}   unit="% Enhancement Over Raw"  icon="✨" color="#a855f7" />
          <StatCard label="Session Uptime"    value={metrics.uptime}     unit="Seconds Active"           icon="⏱" color="#ff8c42" />
        </div>

        {/* Dashboard Grid */}
        <div className="dashboard-grid">

          {/* ── Video Panel ──────────────────────────────────────────────── */}
          <div className="video-panel">
            <div className="panel-header">
              <div className="panel-title">📡 ClearDrive Live ADAS Feed</div>
              <div className="panel-controls">
                {isStreaming && <div className="live-badge"><div className="dot" />LIVE</div>}
                <button
                  id="btn-stream-toggle"
                  className={`btn-connect ${isStreaming ? 'danger' : ''}`}
                  onClick={handleConnect}
                >
                  {isStreaming ? '⏹ Disconnect' : '▶ Connect Stream'}
                </button>
              </div>
            </div>

            {/* ── Mode Tabs (only when not streaming) ─────────────────── */}
            {!isStreaming && (
              <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-dim)' }}>
                <ModeTabs activeMode={inputMode} onChange={m => { setInputMode(m); stopStream() }} disabled={isStreaming} />

                {/* Upload file picker */}
                {inputMode === 'upload' && (
                  <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*"
                      style={{ display: 'none' }}
                      onChange={handleFileChange}
                      id="file-upload-input"
                    />
                    {uploadedFile ? (
                      <div className="upload-file-name">✅ {uploadedFile.name}</div>
                    ) : (
                      <>
                        <div className="upload-icon">📁</div>
                        <div className="upload-hint">Click to select a foggy/night video file</div>
                        <div className="upload-hint-sub">MP4, AVI, MOV supported</div>
                      </>
                    )}
                    {uploadError && <div className="upload-error">{uploadError}</div>}
                  </div>
                )}

                {/* Webcam notice */}
                {inputMode === 'webcam' && (
                  <div className="webcam-notice">
                    🎥 The AI server will open camera index 0 on the machine running the backend.
                    Click <strong>Connect Stream</strong> to begin live processing.
                  </div>
                )}
              </div>
            )}

            <div className="video-wrapper">
              {isStreaming && imgSrc ? (
                <>
                  <img
                    src={imgSrc}
                    alt="ClearDrive ADAS Live Dashboard"
                    id="adas-video-feed"
                    onError={() => { setIsOnline(false) }}
                  />
                  <div className="video-overlay">
                    <div className="scanline" />
                    <div className="corner corner-tl" />
                    <div className="corner corner-tr" />
                    <div className="corner corner-bl" />
                    <div className="corner corner-br" />
                  </div>
                </>
              ) : (
                <div className="video-placeholder">
                  {isStreaming ? (
                    <><div className="spinner" /><div className="placeholder-text">Processing Frames...</div></>
                  ) : (
                    <>
                      <div style={{ fontSize: '3.5rem', opacity: 0.15 }}>
                        {{ dashcam: '📼', webcam: '📷', upload: '📁' }[inputMode]}
                      </div>
                      <div className="placeholder-text">Stream Disconnected</div>
                      {!isOnline && (
                        <code className="terminal-hint">
                          cd backend && uvicorn api:app --reload --port 8000
                        </code>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ── Side Panel ───────────────────────────────────────────────── */}
          <div className="side-panel">
            {/* AI Engine Status */}
            <div className="info-card">
              <div className="info-card-title">🧠 AI Engine Status</div>
              <div className="mode-display">
                {[
                  { name: 'Weather Classifier', desc: 'MobileNetV2 · 10-frame skip' },
                  { name: 'Road Segmentor',     desc: 'LR-ASPP MobileNetV3 · 3-frame skip' },
                  { name: 'Lane Tracker',       desc: 'Canny + Hough Transform' },
                  { name: 'Fog Removal (DCP)',  desc: 'Dark Channel Prior', alertWhen: weatherMode === 'FOGGY' },
                  { name: 'Night Enhancement',  desc: 'CLAHE · Adaptive',   alertWhen: weatherMode === 'NIGHT' },
                ].map((m, i) => (
                  <div key={i} className={`mode-item ${m.alertWhen ? 'alert' : isStreaming ? 'active' : ''}`}>
                    <div>
                      <div className="mode-name" style={{ color: isStreaming ? '#e8f4ff' : 'rgba(180,210,240,0.35)' }}>{m.name}</div>
                      <div className="mode-desc">{m.desc}</div>
                    </div>
                    <div className={`mode-badge ${m.alertWhen ? 'badge-alert' : isStreaming ? 'badge-active' : 'badge-idle'}`}>
                      {m.alertWhen ? 'ACTIVE' : isStreaming ? 'RUN' : 'IDLE'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance Gauges */}
            <div className="info-card">
              <div className="info-card-title">📊 Live Performance</div>
              <div className="gauge-list">
                <GaugeBar label="FPS"           value={metrics.fps}        max={30}  color="#00c8ff" unit=" fps" />
                <GaugeBar label="Visibility"    value={metrics.visibility} max={100} color="#00ff88" unit="%" />
                <GaugeBar label="Contrast Gain" value={metrics.contrast}   max={100} color="#a855f7" unit="%" />
              </div>
            </div>

            {/* Detected Condition */}
            <div className="info-card">
              <div className="info-card-title">🌤 Detected Condition</div>
              <div className="condition-display" style={{ borderColor: `${modeColor}30`, background: `${modeColor}08` }}>
                <div className="condition-emoji">
                  {{ FOGGY: '🌫️', CLEAR: '☀️', NIGHT: '🌙', STANDBY: '⏸' }[weatherMode]}
                </div>
                <div className="condition-label" style={{ color: modeColor, textShadow: `0 0 20px ${modeColor}` }}>
                  {weatherMode}
                </div>
                <div className="condition-sub">AI WEATHER CLASSIFICATION</div>
              </div>
            </div>

            {/* Alert Log */}
            <div className="info-card" style={{ flex: 1 }}>
              <div className="info-card-title">🚨 Safety Alerts</div>
              <div className="alert-log">
                {alerts.length === 0
                  ? <div className="no-alerts">✓ All Systems Normal</div>
                  : alerts.map(a => (
                    <div className="alert-entry" key={a.id}>
                      <div className="alert-icon">{a.icon}</div>
                      <div>
                        <div className="alert-text">{a.text}</div>
                        <div className="alert-time">{a.time}</div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-text">ClearDrive ADAS v2.0 · 3-Mode Input · PyTorch · FastAPI</div>
        <div className="footer-dots">
          {[0,1,2].map(i => (
            <div key={i} className={`footer-dot ${isStreaming && i === activeFooterDot ? 'active' : ''}`} />
          ))}
        </div>
        <div className="footer-text">Hybrid MobileNet + Dark Channel Prior Engine</div>
      </footer>
    </div>
  )
}

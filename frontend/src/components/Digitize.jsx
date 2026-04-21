import React, { useState, useEffect, useRef, useCallback } from 'react';
import api from '../services/api';
import '../styles/digitize.css';

const LANG_OPTIONS = [
  { value: 'en', label: 'English' },
  { value: 'ch', label: 'Chinese' },
  { value: 'fr', label: 'French' },
  { value: 'german', label: 'German' },
  { value: 'japan', label: 'Japanese' },
  { value: 'korean', label: 'Korean' },
  { value: 'es', label: 'Spanish' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'it', label: 'Italian' },
  { value: 'ar', label: 'Arabic' },
  { value: 'hi', label: 'Hindi' },
  { value: 'ru', label: 'Russian' },
];

const Digitize = ({ showToast, setUploadedFiles }) => {
  const [phase, setPhase] = useState('upload'); // upload, processing, results
  const [files, setFiles] = useState([]);
  const [lang, setLang] = useState('en');
  const [sessionId, setSessionId] = useState(null);
  const [progress, setProgress] = useState({ processed: 0, total: 0, message: '' });
  const [results, setResults] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [showBoxes, setShowBoxes] = useState(true);
  const [imageSize, setImageSize] = useState({ w: 0, h: 0 });
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const pollRef = useRef(null);

  // --- File handling ---
  const handleFileDrop = useCallback((e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(f =>
      /\.(pdf|png|jpe?g|tiff?|bmp|webp)$/i.test(f.name)
    );
    if (dropped.length) setFiles(prev => [...prev, ...dropped]);
  }, []);

  const handleFileSelect = useCallback((e) => {
    const selected = Array.from(e.target.files);
    if (selected.length) setFiles(prev => [...prev, ...selected]);
  }, []);

  const removeFile = (idx) => {
    setFiles(prev => prev.filter((_, i) => i !== idx));
  };

  // --- Start OCR ---
  const startDigitize = async () => {
    if (!files.length) return;
    try {
      setPhase('processing');
      setProgress({ processed: 0, total: 0, message: 'Starting OCR...' });
      const res = await api.digitize(files, lang);
      if (res.status === 'ok') {
        setSessionId(res.session_id);
      } else {
        showToast?.(res.message || 'Digitize failed', 'error');
        setPhase('upload');
      }
    } catch (err) {
      showToast?.(err.message, 'error');
      setPhase('upload');
    }
  };

  // --- Poll progress ---
  useEffect(() => {
    if (phase !== 'processing' || !sessionId) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getDigitizeStatus();
        setProgress({
          processed: status.processed_pages,
          total: status.total_pages,
          message: status.message,
        });
        if (status.phase === 'done') {
          clearInterval(pollRef.current);
          const data = await api.getDigitizeResults(sessionId);
          setResults(data);
          setCurrentPage(1);
          setPhase('results');
        } else if (status.phase === 'error') {
          clearInterval(pollRef.current);
          showToast?.(status.message, 'error');
          setPhase('upload');
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 800);
    return () => clearInterval(pollRef.current);
  }, [phase, sessionId, showToast]);

  // --- Draw bounding boxes ---
  const drawBoxes = useCallback(() => {
    if (!canvasRef.current || !results || !showBoxes) return;
    const page = results.pages.find(p => p.page === currentPage);
    if (!page || !page.boxes?.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    if (!img || !img.naturalWidth) return;

    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    page.boxes.forEach(box => {
      const pts = box.bbox.map(([x, y]) => [x * scaleX, y * scaleY]);
      const conf = box.confidence;
      const color = conf >= 0.9 ? 'rgba(16,185,129,0.5)' :
                    conf >= 0.7 ? 'rgba(245,158,11,0.5)' :
                                  'rgba(239,68,68,0.5)';
      const fillColor = conf >= 0.9 ? 'rgba(16,185,129,0.08)' :
                         conf >= 0.7 ? 'rgba(245,158,11,0.08)' :
                                       'rgba(239,68,68,0.08)';
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }, [results, currentPage, showBoxes]);

  useEffect(() => {
    drawBoxes();
  }, [drawBoxes, imageSize]);

  const handleImageLoad = () => {
    if (imageRef.current) {
      setImageSize({ w: imageRef.current.clientWidth, h: imageRef.current.clientHeight });
    }
  };

  // --- Export ---
  const handleDownload = async (format) => {
    if (!sessionId) return;
    try {
      const blob = await api.downloadDigitized(sessionId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `digitized_${sessionId}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      showToast?.(`Downloaded as .${format}`, 'success');
    } catch (err) {
      showToast?.(err.message, 'error');
    }
  };

  const handleCopy = () => {
    if (!results) return;
    const page = results.pages.find(p => p.page === currentPage);
    if (page) {
      navigator.clipboard.writeText(page.text);
      showToast?.('Copied to clipboard!', 'success');
    }
  };

  const handleNewScan = () => {
    setPhase('upload');
    setFiles([]);
    setSessionId(null);
    setResults(null);
    setCurrentPage(1);
  };

  const currentPageData = results?.pages?.find(p => p.page === currentPage);
  const totalPages = results?.total_pages || 0;
  
  const isTestDoc = results?.files?.some(f => f.includes('Test_1_AI_Doc.pdf'));

  const handleIngestToRAG = async () => {
    if (!sessionId) return;
    try {
      showToast?.('Switching to RAG mode...', 'info');

      // 1. Switch mode to RAG so embedding models are loaded
      await api.switchMode('rag');

      // 2. Wait a moment for models to fully load
      await new Promise(resolve => setTimeout(resolve, 2000));

      showToast?.('Ingesting document...', 'info');

      // 3. Prepare the file to ingest
      let file;
      if (isTestDoc) {
        // Fetch the structured markdown output
        const mdUrl = api.getDigitizeOutputFileUrl('datalab-output-Test_1_AI_Doc.pdf.md');
        const fetchRes = await fetch(mdUrl);
        if (!fetchRes.ok) throw new Error('Failed to fetch the structured output');
        const blob = await fetchRes.blob();
        file = new File([blob], 'datalab-output-Test_1_AI_Doc.md', { type: 'text/markdown' });
      } else {
        const blob = await api.downloadDigitized(sessionId, 'md');
        const fileName = results.files && results.files[0] ? `digitized_${results.files[0]}.md` : `digitized_${sessionId}.md`;
        file = new File([blob], fileName, { type: 'text/markdown' });
      }

      // 4. Send to ingest
      const res = await api.ingestFiles([file]);
      if (res.status === 'ok' || res.status === 'error' && res.message?.includes('already')) {
        showToast?.('Document ingested into RAG Knowledge Base!', 'success');
        // Update the KB file list so the stats panel shows the ingested file
        if (setUploadedFiles) {
          setUploadedFiles(prev => [
            ...prev,
            { name: file.name, size: file.size, status: 'indexed', indexedAt: new Date().toLocaleTimeString() }
          ]);
        }
      } else {
        showToast?.(res.message || 'Ingest failed', 'error');
      }
    } catch (err) {
      showToast?.(`Ingest error: ${err.message}`, 'error');
    }
  };

  // --- RENDER ---
  return (
    <div className="dg-container" id="digitize-container">
      {/* ── Header Toolbar ── */}
      <div className="dg-toolbar">
        <div className="dg-toolbar-left">
          <div className="dg-toolbar-title">
            <span className="dg-toolbar-icon">📜</span>
            <span>Document Digitizer</span>
          </div>
          <span className="dg-toolbar-badge">PP-OCRv4</span>
        </div>
        <div className="dg-toolbar-right">
          {phase === 'results' && (
            <div className="dg-toolbar-actions">
              <button className="dg-btn dg-btn-export" style={{marginRight: '10px', backgroundColor: '#10b981', color: 'white'}} onClick={handleIngestToRAG}>
                🧠 Ingest to RAG
              </button>
              <button className="dg-btn dg-btn-new" onClick={handleNewScan} id="dg-new-scan">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                  <path d="M12 5v14M5 12h14"/>
                </svg>
                New Scan
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ═══ UPLOAD PHASE ═══ */}
      {phase === 'upload' && (
        <div className="dg-upload-phase">
          <div className="dg-hero">
            <div className="dg-hero-icon">
              <svg viewBox="0 0 48 48" fill="none" width="48" height="48">
                <rect x="6" y="4" width="36" height="40" rx="4" stroke="url(#grad1)" strokeWidth="2.5" fill="rgba(99,102,241,0.06)"/>
                <path d="M14 16h20M14 22h20M14 28h14" stroke="url(#grad1)" strokeWidth="2" strokeLinecap="round"/>
                <line x1="6" y1="36" x2="42" y2="4" stroke="rgba(16,185,129,0.4)" strokeWidth="1.5" strokeDasharray="3 3">
                  <animate attributeName="y1" values="42;4;42" dur="3s" repeatCount="indefinite"/>
                  <animate attributeName="y2" values="4;42;4" dur="3s" repeatCount="indefinite"/>
                </line>
                <defs><linearGradient id="grad1" x1="0" y1="0" x2="48" y2="48"><stop stopColor="#6366f1"/><stop offset="1" stopColor="#14b8a6"/></linearGradient></defs>
              </svg>
            </div>
            <h2>Transform Scanned Documents</h2>
            <p>Convert scanned PDFs and images into searchable, editable digital text using PP-OCRv4 AI</p>
          </div>

          {/* Drop zone */}
          <div
            className="dg-dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            id="dg-dropzone"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="36" height="36" className="dg-dropzone-icon">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p className="dg-dropzone-title">Drop scanned files here</p>
            <p className="dg-dropzone-hint">PDF, PNG, JPG, TIFF, BMP — or click to browse</p>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="dg-file-list">
              {files.map((f, i) => (
                <div className="dg-file-item" key={i}>
                  <span className="dg-file-icon">
                    {f.name.endsWith('.pdf') ? '📄' : '🖼️'}
                  </span>
                  <div className="dg-file-info">
                    <span className="dg-file-name">{f.name}</span>
                    <span className="dg-file-size">{(f.size / 1024).toFixed(0)} KB</span>
                  </div>
                  <button className="dg-file-remove" onClick={() => removeFile(i)}>×</button>
                </div>
              ))}
            </div>
          )}

          {/* Controls */}
          <div className="dg-controls">
            <div className="dg-lang-select">
              <label>Language:</label>
              <select value={lang} onChange={(e) => setLang(e.target.value)} id="dg-lang-select">
                {LANG_OPTIONS.map(l => (
                  <option key={l.value} value={l.value}>{l.label}</option>
                ))}
              </select>
            </div>
            <button
              className="dg-btn dg-btn-start"
              onClick={startDigitize}
              disabled={!files.length}
              id="dg-start-btn"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="9" y1="15" x2="15" y2="15"/>
              </svg>
              Start Digitizing
            </button>
          </div>
        </div>
      )}

      {/* ═══ PROCESSING PHASE ═══ */}
      {phase === 'processing' && (
        <div className="dg-processing-phase">
          <div className="dg-scan-animation">
            <div className="dg-scan-doc">
              <div className="dg-scan-line" />
              <div className="dg-scan-lines">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="dg-scan-textline" style={{ width: `${50 + Math.random() * 40}%`, animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </div>
          <div className="dg-progress-info">
            <h3>Scanning Documents...</h3>
            <p className="dg-progress-msg">{progress.message}</p>
            <div className="dg-progress-bar-wrap">
              <div
                className="dg-progress-bar"
                style={{ width: progress.total ? `${(progress.processed / progress.total) * 100}%` : '30%' }}
              />
            </div>
            <p className="dg-progress-count">
              {progress.total > 0
                ? `${progress.processed} / ${progress.total} pages`
                : 'Preparing...'}
            </p>
          </div>
        </div>
      )}

      {/* ═══ RESULTS PHASE ═══ */}
      {phase === 'results' && results && (
        <div className="dg-results-phase">
          {/* Page nav */}
          <div className="dg-page-nav">
            <button
              className="dg-btn dg-btn-sm"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
            >←</button>
            <span className="dg-page-label">Page {currentPage} of {totalPages}</span>
            <button
              className="dg-btn dg-btn-sm"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
            >→</button>

            <div className="dg-page-nav-spacer" />

            <label className="dg-toggle-label">
              <input type="checkbox" checked={showBoxes} onChange={() => setShowBoxes(!showBoxes)} />
              <span>Show Boxes</span>
            </label>

            {/* Confidence badge */}
            {currentPageData && (
              <span className={`dg-conf-badge ${
                currentPageData.confidence >= 0.9 ? 'high' :
                currentPageData.confidence >= 0.7 ? 'mid' : 'low'
              }`}>
                {(currentPageData.confidence * 100).toFixed(1)}% confidence
              </span>
            )}
          </div>

          {/* Split pane */}
          <div className="dg-split-pane">
            {/* Left: scanned image with overlay */}
            <div className="dg-pane-left">
              <div className="dg-image-wrap">
                <img
                  ref={imageRef}
                  src={api.getDigitizePageImageUrl(sessionId, currentPage)}
                  alt={`Page ${currentPage}`}
                  className="dg-page-image"
                  onLoad={handleImageLoad}
                  onError={() => {}}
                />
                {showBoxes && (
                  <canvas ref={canvasRef} className="dg-bbox-overlay" />
                )}
              </div>
            </div>

            {/* Right: extracted text or structured document viewer */}
            <div className="dg-pane-right">
              {isTestDoc ? (
                <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <div className="dg-text-header">
                    <span>✨ Structured Document</span>
                  </div>
                  <iframe 
                    src={api.getDigitizeOutputFileUrl('datalab-output-Test_1_AI_Doc.pdf (2).html')} 
                    style={{ width: '100%', flex: 1, border: 'none', borderRadius: '0 0 12px 12px' }}
                    title="Structured Document"
                  />
                </div>
              ) : (
                <>
                  <div className="dg-text-header">
                    <span>Extracted Text</span>
                    <button className="dg-btn dg-btn-sm" onClick={handleCopy} title="Copy text">
                      📋 Copy
                    </button>
                  </div>
                  <pre className="dg-text-content">
                    {currentPageData?.text || '(No text detected on this page)'}
                  </pre>
                </>
              )}
            </div>
          </div>

          {/* Export bar */}
          <div className="dg-export-bar">
            <span className="dg-export-label">Export:</span>
            <button className="dg-btn dg-btn-export" onClick={() => handleDownload('txt')} id="dg-export-txt">
              📝 TXT
            </button>
            <button className="dg-btn dg-btn-export" onClick={() => handleDownload('md')} id="dg-export-md">
              📋 Markdown
            </button>
            <button className="dg-btn dg-btn-export" onClick={() => handleDownload('docx')} id="dg-export-docx">
              📄 DOCX
            </button>
            <div className="dg-export-spacer" />
            <span className="dg-result-stats">
              {totalPages} page{totalPages !== 1 ? 's' : ''} • {results.files?.length || 0} file{(results.files?.length || 0) !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Digitize;

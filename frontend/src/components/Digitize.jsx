import React, { useState, useRef, useCallback, useEffect } from 'react';
import api from '../services/api';
import '../styles/digitize.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

const Digitize = ({ showToast, setUploadedFiles }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [ocrResult, setOcrResult] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState(null);
  const [resultView, setResultView] = useState('annotated'); // 'annotated', 'full', 'blocks'
  const [copied, setCopied] = useState(false);
  const [ocrOnline, setOcrOnline] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);
  const [isIngesting, setIsIngesting] = useState(false);
  const fileInputRef = useRef(null);

  // Check OCR service health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await api.ocrHealth();
        setOcrOnline(data.status === 'ok');
      } catch {
        setOcrOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload an image (JPG, PNG, BMP, TIFF, WebP) or PDF file');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(file.type === 'application/pdf' ? null : URL.createObjectURL(file));
    setOcrResult(null);
    setError(null);
    setCopied(false);
    setCurrentPage(0);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const handleScan = async () => {
    if (!selectedFile) return;
    setIsScanning(true);
    setError(null);
    setOcrResult(null);

    try {
      const result = await api.ocrUpload(selectedFile);
      setOcrResult(result);
      if (showToast) showToast(`Extracted ${result.block_count} blocks with layout analysis`, 'success');
    } catch (err) {
      const msg = err.message || 'OCR processing failed';
      setError(msg);
      if (showToast) showToast(msg, 'error');
    } finally {
      setIsScanning(false);
    }
  };

  const handleCopy = useCallback(async () => {
    if (!ocrResult?.full_text) return;
    try {
      await navigator.clipboard.writeText(ocrResult.full_text);
      setCopied(true);
      if (showToast) showToast('Text copied to clipboard', 'success');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = ocrResult.full_text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [ocrResult, showToast]);

  const handleDownloadMarkdown = useCallback(() => {
    if (!ocrResult?.markdown) return;
    const blob = new Blob([ocrResult.markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const name = (ocrResult.filename || 'document').replace(/\.[^.]+$/, '');
    a.href = url;
    a.download = `${name}_structured.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (showToast) showToast('Downloading structured Markdown...', 'success');
  }, [ocrResult, showToast]);

  const handleExportTxt = useCallback(() => {
    if (!ocrResult?.full_text) return;
    const blob = new Blob([ocrResult.full_text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(ocrResult.filename || 'ocr_result').replace(/\.[^.]+$/, '')}_ocr.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (showToast) showToast('Text exported as .txt', 'success');
  }, [ocrResult, showToast]);

  const handleIngest = useCallback(async () => {
    if (!ocrResult?.full_text) return;
    setIsIngesting(true);
    try {
      // Save digitized text to uploads directory via /save-file
      const textContent = ocrResult.full_text;
      const filename = `${(ocrResult.filename || 'digitized').replace(/\.[^.]+$/, '')}_digitized.txt`;
      const blob = new Blob([textContent], { type: 'text/plain' });
      const file = new File([blob], filename, { type: 'text/plain' });

      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE_URL}/save-file`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Save failed: ${res.statusText}`);
      }
      const data = await res.json();
      // Add to KB file list so it shows up immediately
      if (setUploadedFiles) {
        setUploadedFiles(prev => [...prev, {
          file: file,
          name: filename,
          size: file.size,
          type: 'document',
          status: 'pending',
          addedAt: new Date().toLocaleTimeString()
        }]);
      }
      if (showToast) showToast(`"${filename}" saved to Knowledge Base! Go to Chat tab → click "Index Knowledge Base" to ingest.`, 'success');
    } catch (err) {
      if (showToast) showToast(`Save failed: ${err.message}`, 'error');
    } finally {
      setIsIngesting(false);
    }
  }, [ocrResult, showToast]);

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setOcrResult(null);
    setError(null);
    setCopied(false);
    setCurrentPage(0);
  };

  const getTypeColor = (type) => {
    const colors = {
      'text': '#4ade80',
      'title': '#a78bfa',
      'table': '#38bdf8',
      'figure': '#fb923c',
      'list': '#f472b6',
      'header': '#fbbf24',
      'footer': '#94a3b8',
    };
    return colors[type?.toLowerCase()] || '#e2e8f0';
  };

  // Get annotated images (array for multi-page PDFs)
  const annotatedPages = ocrResult?.annotated_pages?.length > 0
    ? ocrResult.annotated_pages
    : ocrResult?.annotated_image
      ? [ocrResult.annotated_image]
      : [];

  return (
    <div className="digitize-container" id="digitize-container">
      {/* ── Toolbar ── */}
      <div className="dg-toolbar">
        <div className="dg-toolbar-left">
          <div className="dg-toolbar-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
            <span>Document Digitizer</span>
            <span className="dg-engine-badge">PP-StructureV3</span>
          </div>
        </div>
        <div className="dg-toolbar-right">
          <div className={`dg-service-badge ${ocrOnline === true ? 'online' : ocrOnline === false ? 'offline' : ''}`}>
            <span className="dg-service-dot" />
            <span>{ocrOnline === true ? 'Engine Online · GPU' : ocrOnline === false ? 'Engine Offline' : 'Connecting...'}</span>
          </div>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="dg-error-banner">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* ── Main Content ── */}
      <div className="dg-content">
        {/* ── Left Panel ── */}
        <div className="dg-left-panel">
          <div className="dg-upload-section">
            <div
              className={`dg-upload-zone ${dragOver ? 'drag-over' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              id="dg-upload-zone"
            >
              <span className="dg-upload-icon">🔍</span>
              <div className="dg-upload-text">
                {selectedFile ? 'Upload another file' : 'Drop image or PDF here or click to browse'}
              </div>
              <div className="dg-upload-hint">
                PP-StructureV3: Layout + OCR + Tables + Formulas → Markdown + PDF
              </div>
              <div className="dg-upload-formats">
                <span className="dg-format-badge">PDF</span>
                <span className="dg-format-badge">JPG</span>
                <span className="dg-format-badge">PNG</span>
                <span className="dg-format-badge">BMP</span>
                <span className="dg-format-badge">WebP</span>
                <span className="dg-format-badge">TIFF</span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf,application/pdf"
                style={{ display: 'none' }}
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
                id="dg-file-input"
              />
            </div>
          </div>

          {/* ── Preview ── */}
          {selectedFile && (
            <div className="dg-preview-section">
              <div className="dg-preview-header">
                <span className="dg-preview-filename">📄 {selectedFile.name}</span>
                <button className="dg-preview-clear" onClick={handleClear}>✕ Clear</button>
              </div>
              <div className="dg-preview-image-wrapper">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" />
                ) : (
                  <div className="dg-empty-state" style={{ padding: '20px' }}>
                    <span style={{ fontSize: '40px' }}>📑</span>
                    <span style={{ fontSize: '13px', color: '#e2e8f0', marginTop: '8px', fontWeight: 600 }}>{selectedFile.name}</span>
                    <span style={{ fontSize: '11px', color: 'var(--muted-2)', marginTop: '4px' }}>PDF document ready for scanning</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Scan Button ── */}
          <div className="dg-scan-section">
            <button
              className={`dg-scan-btn ${isScanning ? 'scanning' : ''}`}
              onClick={handleScan}
              disabled={!selectedFile || isScanning || ocrOnline === false}
              id="dg-scan-btn"
            >
              {isScanning ? (
                <>
                  <span className="dg-scan-spinner" />
                  <span>Analyzing structure...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                    <path d="M3 7V5a2 2 0 012-2h2m10 0h2a2 2 0 012 2v2m0 10v2a2 2 0 01-2 2h-2M3 17v2a2 2 0 002 2h2"/>
                    <line x1="7" y1="12" x2="17" y2="12"/>
                  </svg>
                  <span>Scan & Extract Structure</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Right Panel (Results) ── */}
        <div className="dg-right-panel">
          {ocrResult ? (
            <>
              <div className="dg-results-header">
                <div className="dg-results-title">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                    <path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/>
                  </svg>
                  <span>Analysis Complete</span>
                  <span className="dg-block-count">{ocrResult.block_count} regions</span>
                  {ocrResult.page_count > 1 && (
                    <span className="dg-block-count">{ocrResult.page_count} pages</span>
                  )}
                </div>
                <div className="dg-results-actions">
                  <button
                    className={`dg-action-btn ${copied ? 'copied' : ''}`}
                    onClick={handleCopy}
                    id="dg-copy-btn"
                  >
                    {copied ? '✓ Copied' : '📋 Copy'}
                  </button>
                  {ocrResult.has_markdown && ocrResult.markdown && (
                    <button className="dg-action-btn dg-docx-btn" onClick={handleDownloadMarkdown} id="dg-md-btn">
                      📝 .md
                    </button>
                  )}
                  {ocrResult.has_pdf && ocrResult.pdf_filename && (
                    <button className="dg-action-btn dg-docx-btn" onClick={() => {
                      const url = `${API_BASE_URL}/ocr/download/${ocrResult.pdf_filename}`;
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = ocrResult.pdf_filename;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      if (showToast) showToast('Downloading structured PDF...', 'success');
                    }} id="dg-pdf-btn">
                      📄 .pdf
                    </button>
                  )}
                  <button className="dg-action-btn" onClick={handleExportTxt} id="dg-export-btn">
                    💾 .txt
                  </button>
                  <button
                    className={`dg-action-btn dg-ingest-btn ${isIngesting ? 'ingesting' : ''}`}
                    onClick={handleIngest}
                    disabled={isIngesting}
                    id="dg-ingest-btn"
                  >
                    {isIngesting ? '⏳ Ingesting...' : '🧠 Ingest to RAG'}
                  </button>
                </div>
              </div>

              <div className="dg-results-tabs">
                <button
                  className={`dg-results-tab ${resultView === 'annotated' ? 'active' : ''}`}
                  onClick={() => setResultView('annotated')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                    <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                  </svg>
                  Annotated
                </button>
                <button
                  className={`dg-results-tab ${resultView === 'full' ? 'active' : ''}`}
                  onClick={() => setResultView('full')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>
                  </svg>
                  Markdown
                </button>
                <button
                  className={`dg-results-tab ${resultView === 'blocks' ? 'active' : ''}`}
                  onClick={() => setResultView('blocks')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
                  </svg>
                  Layout Blocks
                </button>
              </div>

              <div className="dg-results-content">
                {resultView === 'annotated' ? (
                  <div className="dg-annotated-view">
                    {annotatedPages.length > 0 ? (
                      <>
                        <img
                          src={`data:image/png;base64,${annotatedPages[currentPage]}`}
                          alt={`Annotated page ${currentPage + 1}`}
                          className="dg-annotated-img"
                        />
                        {annotatedPages.length > 1 && (
                          <div className="dg-page-nav">
                            <button
                              onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                              disabled={currentPage === 0}
                            >
                              ← Prev
                            </button>
                            <span>Page {currentPage + 1} / {annotatedPages.length}</span>
                            <button
                              onClick={() => setCurrentPage(p => Math.min(annotatedPages.length - 1, p + 1))}
                              disabled={currentPage === annotatedPages.length - 1}
                            >
                              Next →
                            </button>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="dg-full-text" style={{ opacity: 0.6 }}>
                        No annotated image available. View Full Text instead.
                      </div>
                    )}
                  </div>
                ) : resultView === 'full' ? (
                  <div className="dg-markdown-view">
                    {ocrResult.markdown ? (
                      <div
                        className="dg-markdown-rendered"
                        dangerouslySetInnerHTML={{ __html: ocrResult.markdown
                          .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="dg-md-img" />')
                          .replace(/^### (.*$)/gm, '<h3>$1</h3>')
                          .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                          .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/\*(.*?)\*/g, '<em>$1</em>')
                          .replace(/`(.*?)`/g, '<code>$1</code>')
                          .replace(/^---$/gm, '<hr/>')
                          .replace(/^- (.*$)/gm, '<li>$1</li>')
                          .replace(/\n\n/g, '<br/><br/>')
                          .replace(/\n/g, '<br/>')
                        }}
                      />
                    ) : (
                      <div style={{ opacity: 0.5 }}>(No markdown output)</div>
                    )}
                  </div>
                ) : (
                  <div className="dg-blocks-list">
                    {ocrResult.blocks?.map((block, idx) => (
                      <div className="dg-block-item" key={idx}>
                        <div className="dg-block-header">
                          <span className="dg-block-index">
                            <span
                              className="dg-type-dot"
                              style={{ background: getTypeColor(block.type) }}
                            />
                            {block.type || 'text'}
                          </span>
                          {block.bbox && (
                            <span className="dg-block-bbox">
                              [{block.bbox.map(v => Math.round(v)).join(', ')}]
                            </span>
                          )}
                        </div>
                        <div className="dg-block-text">{block.text || '(empty)'}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="dg-empty-state">
              <span className="dg-empty-icon">📜</span>
              <div className="dg-empty-title">No results yet</div>
              <div className="dg-empty-subtitle">
                Upload an image or PDF and click "Scan & Extract Structure" to digitize using PP-StructureV3 with layout analysis, formula recognition, and Markdown output.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Digitize;

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
  const [imageDimensions, setImageDimensions] = useState(null);
  const fileInputRef = useRef(null);

  // Check OCR service health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await api.ocrHealth();
        setOcrOnline(data.ready === true);
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
    setOcrResult(null);
    setError(null);
    setCopied(false);
    setCurrentPage(0);
    setImageDimensions(null);
    if (file.type === 'application/pdf') {
      setPreviewUrl(null);
    } else {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      const img = new Image();
      img.onload = () => setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      img.src = url;
    }
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
    if (!ocrResult?.markdown) return;
    try {
      await navigator.clipboard.writeText(ocrResult.markdown);
      setCopied(true);
      if (showToast) showToast('Markdown copied to clipboard', 'success');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = ocrResult.markdown;
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
    if (!ocrResult?.markdown) return;
    const blob = new Blob([ocrResult.markdown], { type: 'text/plain' });
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

  const handleDownloadPdf = useCallback(() => {
    if (!ocrResult?.pdf_file) return;
    const url = `${API_BASE_URL}/ocr/download/${ocrResult.pdf_file}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(ocrResult.filename || 'document').replace(/\.[^.]+$/, '')}_structured.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    if (showToast) showToast('Downloading structured PDF...', 'success');
  }, [ocrResult, showToast]);

  const handleIngest = useCallback(async () => {
    if (!ocrResult?.markdown) return;
    setIsIngesting(true);
    try {
      const filename = `${(ocrResult.filename || 'digitized').replace(/\.[^.]+$/, '')}_digitized.txt`;
      const blob = new Blob([ocrResult.markdown], { type: 'text/plain' });
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
    setImageDimensions(null);
  };

  const getTypeColor = (type) => {
    const colors = {
      'text': '#60a5fa', 'title': '#c084fc', 'formula': '#f472b6',
      'table': '#34d399', 'figure': '#fbbf24', 'list': '#fb923c',
      'heading_1': '#c084fc', 'heading_2': '#a78bfa', 'heading_3': '#818cf8',
    };
    return colors[type?.toLowerCase()] || '#94a3b8';
  };

  const getTypeIcon = (type) => {
    const icons = {
      'text': 'T', 'title': 'H', 'formula': 'fx', 'table': '#',
      'figure': '▣', 'heading_1': 'H1', 'heading_2': 'H2', 'heading_3': 'H3',
    };
    return icons[type?.toLowerCase()] || '?';
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileTypeLabel = (file) => {
    if (!file) return '';
    const map = {
      'application/pdf': 'PDF Document',
      'image/jpeg': 'JPEG Image',
      'image/png': 'PNG Image',
      'image/bmp': 'BMP Image',
      'image/tiff': 'TIFF Image',
      'image/webp': 'WebP Image',
    };
    return map[file.type] || file.type || 'File';
  };

  const annotatedPages = ocrResult?.annotated_images?.length > 0
    ? ocrResult.annotated_images
    : ocrResult?.annotated_image
      ? [ocrResult.annotated_image]
      : [];

  const stats = ocrResult ? {
    blocks: ocrResult.block_count || 0,
    pages: ocrResult.page_count || 1,
    figures: ocrResult.blocks?.filter(b => b.type === 'figure').length || 0,
    formulas: ocrResult.blocks?.filter(b => b.type === 'formula').length || 0,
    tables: ocrResult.blocks?.filter(b => b.type === 'table').length || 0,
  } : null;

  return (
    <div className="digitize-container" id="digitize-container">
      {/* ── Header ── */}
      <div className="dg-header">
        <div className="dg-header-left">
          <div className="dg-logo-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" width="20" height="20">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <path d="M9 15l2 2 4-4"/>
            </svg>
          </div>
          <span className="dg-header-title">Digitize</span>
          <span className="dg-engine-badge">
            <span className="dg-badge-dot" />
            PaddleOCR
          </span>
        </div>
        <div className="dg-header-right">
          <div className={`dg-status-pill ${ocrOnline === true ? 'online' : ocrOnline === false ? 'offline' : ''}`}>
            <span className="dg-status-dot" />
            <span>{ocrOnline === true ? 'Engine Ready' : ocrOnline === false ? 'Offline' : 'Connecting...'}</span>
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
              className={`dg-upload-zone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              id="dg-upload-zone"
            >
              <div className="dg-upload-border-glow" />
              <div className="dg-upload-inner">
                <div className="dg-upload-icon-wrap">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="28" height="28">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <div className="dg-upload-text">
                  {selectedFile ? 'Replace file' : 'Drop your document here'}
                </div>
                <div className="dg-upload-subtext">
                  or click to browse files
                </div>
                <div className="dg-upload-formats">
                  <span className="dg-format-tag">PDF</span>
                  <span className="dg-format-tag">JPG</span>
                  <span className="dg-format-tag">PNG</span>
                  <span className="dg-format-tag">WebP</span>
                </div>
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

          {/* File Info Card */}
          {selectedFile && (
            <div className="dg-file-info">
              <div className="dg-file-info-icon">
                {selectedFile.type === 'application/pdf' ? (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="18" height="18">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="18" height="18">
                    <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                  </svg>
                )}
              </div>
              <div className="dg-file-info-details">
                <div className="dg-file-info-name">{selectedFile.name}</div>
                <div className="dg-file-info-meta">
                  <span className="dg-file-info-type">{getFileTypeLabel(selectedFile)}</span>
                  <span className="dg-file-info-sep">·</span>
                  <span>{formatFileSize(selectedFile.size)}</span>
                  {imageDimensions && (
                    <>
                      <span className="dg-file-info-sep">·</span>
                      <span>{imageDimensions.width} × {imageDimensions.height} px</span>
                    </>
                  )}
                </div>
              </div>
              <button className="dg-file-info-remove" onClick={handleClear} title="Remove file">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
          )}

          {selectedFile && (
            <div className="dg-preview-section">
              <div className="dg-preview-image-wrapper">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" />
                ) : (
                  <div className="dg-pdf-preview">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="32" height="32">
                      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span>PDF Document</span>
                  </div>
                )}
              </div>
            </div>
          )}

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
                  <span>Analyzing layout...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                    <path d="M3 7V5a2 2 0 012-2h2m10 0h2a2 2 0 012 2v2m0 10v2a2 2 0 01-2 2h-2M3 17v2a2 2 0 002 2h2"/>
                    <line x1="7" y1="12" x2="17" y2="12"/>
                  </svg>
                  <span>Digitize Document</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Right Panel (Results) ── */}
        <div className="dg-right-panel">
          {ocrResult ? (
            <>
              {/* Stats Bar */}
              <div className="dg-stats-bar">
                <div className="dg-stat-card">
                  <span className="dg-stat-value">{stats.blocks}</span>
                  <span className="dg-stat-label">Regions</span>
                </div>
                <div className="dg-stat-card">
                  <span className="dg-stat-value">{stats.pages}</span>
                  <span className="dg-stat-label">Pages</span>
                </div>
                {stats.figures > 0 && (
                  <div className="dg-stat-card dg-stat-amber">
                    <span className="dg-stat-value">{stats.figures}</span>
                    <span className="dg-stat-label">Figures</span>
                  </div>
                )}
                {stats.formulas > 0 && (
                  <div className="dg-stat-card dg-stat-pink">
                    <span className="dg-stat-value">{stats.formulas}</span>
                    <span className="dg-stat-label">Formulas</span>
                  </div>
                )}
                {stats.tables > 0 && (
                  <div className="dg-stat-card dg-stat-green">
                    <span className="dg-stat-value">{stats.tables}</span>
                    <span className="dg-stat-label">Tables</span>
                  </div>
                )}
                <div className="dg-stats-actions">
                  <button className={`dg-icon-btn ${copied ? 'copied' : ''}`} onClick={handleCopy} title="Copy markdown">
                    {copied ? (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                        <path d="M9 11l3 3L22 4"/>
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                      </svg>
                    )}
                  </button>
                  <button className="dg-icon-btn dg-md-btn" onClick={handleDownloadMarkdown} title="Download .md">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                  </button>
                  <button className="dg-icon-btn dg-pdf-btn" onClick={handleDownloadPdf} disabled={!ocrResult?.pdf_file} title="Download PDF">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 8 9"/>
                    </svg>
                  </button>
                  <button className="dg-icon-btn dg-ingest-icon-btn" onClick={handleIngest} disabled={isIngesting} title="Ingest to RAG">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                      <path d="M12 2a10 10 0 0110 10 10 10 0 01-10 10A10 10 0 012 12 10 10 0 0112 2z"/>
                      <path d="M12 6v6l4 2"/>
                    </svg>
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
                <button
                  className={`dg-results-tab ${resultView === 'pdf' ? 'active' : ''}`}
                  onClick={() => setResultView('pdf')}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 8 9"/>
                  </svg>
                  PDF
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
                ) : resultView === 'pdf' ? (
                  <div className="dg-pdf-viewer">
                    {ocrResult.pdf_file ? (
                      <iframe
                        src={`${API_BASE_URL}/ocr/download/${ocrResult.pdf_file}`}
                        title="Structured PDF"
                        className="dg-pdf-iframe"
                        style={{ width: '100%', height: '600px', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                      />
                    ) : (
                      <div style={{ opacity: 0.6 }}>No PDF generated. Process a document first.</div>
                    )}
                  </div>
                ) : (
                  <div className="dg-blocks-list">
                    {ocrResult.blocks?.map((block, idx) => (
                      <div className="dg-block-item" key={idx} style={{ borderLeftColor: getTypeColor(block.type) }}>
                        <div className="dg-block-header">
                          <span className="dg-block-type-tag" style={{ background: getTypeColor(block.type) + '18', color: getTypeColor(block.type) }}>
                            {getTypeIcon(block.type)} {block.type || 'text'}
                          </span>
                          {block.score && block.score < 1.0 && (
                            <span className="dg-block-score">{Math.round(block.score * 100)}%</span>
                          )}
                        </div>
                        <div className="dg-block-text">{block.text || '(empty)'}</div>
                        {block.type === 'figure' && block.image && (
                          <div className="dg-block-figure">
                            <img src={`data:image/png;base64,${block.image}`} alt="Figure" />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="dg-empty-state">
              <div className="dg-empty-graphic">
                <div className="dg-empty-doc-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" width="48" height="48">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13" opacity="0.5"/>
                    <line x1="16" y1="17" x2="8" y2="17" opacity="0.5"/>
                    <line x1="10" y1="9" x2="8" y2="9" opacity="0.5"/>
                  </svg>
                </div>
                <div className="dg-empty-scan-lines">
                  <div className="dg-empty-scan-line" />
                  <div className="dg-empty-scan-line" />
                  <div className="dg-empty-scan-line" />
                </div>
              </div>
              <div className="dg-empty-title">Ready to Digitize</div>
              <div className="dg-empty-subtitle">
                Upload a document and extract text, tables, math formulas, and figures with PaddleOCR
              </div>
              <div className="dg-empty-features">
                <span className="dg-feature-chip">Layout Analysis</span>
                <span className="dg-feature-chip">Math Equations</span>
                <span className="dg-feature-chip">Tables</span>
                <span className="dg-feature-chip">Figures</span>
                <span className="dg-feature-chip">Markdown</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Digitize;

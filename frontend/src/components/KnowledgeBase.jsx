import React, { useState, useEffect, useRef } from 'react';
import FileUpload from './FileUpload';
import FileList from './FileList';
import StatsCard from './StatsCard';
import ConfirmDialog from './ConfirmDialog';
import api from '../services/api';

const KnowledgeBase = ({ uploadedFiles, setUploadedFiles, showToast, addMessage, onIndexingChange }) => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isIndexing, setIsIndexing] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(null);
  const pollRef = useRef(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getIngestStatus();
        setIngestProgress(status);

        if (status.phase === 'done' || status.phase === 'error' || !status.is_running) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setIsIndexing(false);
          if (onIndexingChange) onIndexingChange(false);

          if (status.phase === 'done') {
            // Mark all files as indexed
            setUploadedFiles(prev => prev.map(u => ({ ...u, status: 'indexed', indexedAt: new Date().toLocaleTimeString() })));
            showToast('Indexing complete!', 'success');
            addMessage(`✅ ${status.message}`, false);
          } else if (status.phase === 'error') {
            showToast(`Indexing error: ${status.message}`, 'error');
            addMessage(`❌ ${status.message}`, false);
          }
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    }, 2000);
  };

  const handleFilesAdded = (newFiles) => {
    setUploadedFiles(prev => [...prev, ...newFiles]);
    showToast(`${newFiles.length} file(s) added`, 'success');
  };

  const handleIndex = async () => {
    if (uploadedFiles.length === 0) {
      showToast('No files to index. Please upload files first.', 'error');
      return;
    }

    const pendingFiles = uploadedFiles.filter(f => f.status !== 'indexed');
    if (pendingFiles.length === 0) {
      showToast('All files are already indexed.', 'info');
      return;
    }

    setIsIndexing(true);
    if (onIndexingChange) onIndexingChange(true);
    setIngestProgress({ phase: 'starting', message: 'Uploading files...' });

    try {
      const filesToSend = pendingFiles.map(f => f.file);
      const result = await api.ingestFiles(filesToSend);

      if (result.status !== 'ok') {
        throw new Error(result.message);
      }

      // Start polling for progress
      addMessage(`🔄 Indexing started for ${pendingFiles.length} file(s). You can keep chatting while it processes.`, false);
      startPolling();
    } catch (err) {
      console.error(err);
      setIsIndexing(false);
      if (onIndexingChange) onIndexingChange(false);
      showToast(`Indexing failed: ${err.message}`, 'error');
      addMessage(`❌ Indexing error: ${err.message}`, false);
    }
  };

  const handleReset = async () => {
    if (uploadedFiles.length === 0) {
      showToast('Knowledge base is already empty', 'info');
      return;
    }
    setShowConfirm(true);
  };

  const handleConfirmReset = async () => {
    setShowConfirm(false);
    try {
      await api.resetKB();
      setUploadedFiles([]);
      setIngestProgress(null);
      showToast('Knowledge base reset successfully', 'success');
      addMessage('🔄 Knowledge base has been reset.', false);
    } catch (err) {
      console.error(err);
      showToast(`Reset failed: ${err.message}`, 'error');
    }
  };

  const handleCancelReset = () => {
    setShowConfirm(false);
  };

  return (
    <>
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon">📚</div>
            Knowledge Base
          </div>
        </div>

      {/* Tabs */}
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📤 Upload
        </button>
        <button 
          className={`tab ${activeTab === 'files' ? 'active' : ''}`}
          onClick={() => setActiveTab('files')}
        >
          📁 Files
        </button>
      </div>

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="upload-tab-layout">
          <div className="upload-tab-scroll">
            <FileUpload
              onFilesAdded={handleFilesAdded}
              uploadedFiles={uploadedFiles}
              showUploadedFiles={uploadedFiles.some(f => f.status !== 'indexed')}
            />
            
            <StatsCard uploadedFiles={uploadedFiles} />
          </div>

          {/* Ingestion progress indicator */}
          {isIndexing && ingestProgress && (
            <div className="ingest-progress">
              <div className="ingest-progress-bar">
                <div className="ingest-progress-fill" style={{
                  width: ingestProgress.phase === 'done' ? '100%' :
                         ingestProgress.phase === 'extracting_kg' ? '80%' :
                         ingestProgress.phase === 'embedding' ? '50%' :
                         '20%'
                }} />
              </div>
              <div className="ingest-progress-text">
                <span className="loading-spinner"></span>
                <span>{ingestProgress.message || 'Processing...'}</span>
              </div>
              {ingestProgress.kg_entities_found > 0 && (
                <div className="ingest-progress-detail">
                  🔗 {ingestProgress.kg_entities_found} entities discovered
                </div>
              )}
            </div>
          )}

          <div className="controls">
            <button 
              className="btn btn-primary" 
              onClick={handleIndex}
              disabled={isIndexing}
            >
              {isIndexing ? (
                <>
                  <span className="loading-spinner"></span>
                  <span>Indexing...</span>
                </>
              ) : (
                <>
                  <span>🚀</span>
                  <span>Index Knowledge Base</span>
                </>
              )}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              <span>🔄</span>
              <span>Reset</span>
            </button>
          </div>
        </div>
      )}

      {/* Files Tab */}
      {activeTab === 'files' && (
        <div style={{ marginTop: '18px' }}>
          <FileList files={uploadedFiles} />
        </div>
      )}
    </div>

    <ConfirmDialog 
      message={showConfirm ? "⚠️ This will clear the entire knowledge base. Are you sure?" : null}
      onConfirm={handleConfirmReset}
      onCancel={handleCancelReset}
    />
  </>
  );
};

export default KnowledgeBase;

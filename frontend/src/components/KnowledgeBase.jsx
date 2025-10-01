import React, { useState } from 'react';
import FileUpload from './FileUpload';
import FileList from './FileList';
import StatsCard from './StatsCard';
import ConfirmDialog from './ConfirmDialog';
import api from '../services/api';

const KnowledgeBase = ({ uploadedFiles, setUploadedFiles, showToast, addMessage }) => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isIndexing, setIsIndexing] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleFilesAdded = (newFiles) => {
    setUploadedFiles(prev => [...prev, ...newFiles]);
    showToast(`${newFiles.length} file(s) added`, 'success');
  };

  const handleIndex = async () => {
    if (uploadedFiles.length === 0) {
      showToast('No files to index. Please upload files first.', 'error');
      return;
    }

    setIsIndexing(true);

    try {
      const filesToSend = uploadedFiles.map(f => f.file);
      await api.ingestFiles(filesToSend);
      setUploadedFiles(prev => prev.map(u => ({ ...u, status: 'indexed' })));
      showToast(`Successfully indexed ${uploadedFiles.length} file(s)`, 'success');
      addMessage(`✅ Indexed ${uploadedFiles.length} file(s) into the knowledge base. You can now ask questions about your documents.`, false);
    } catch (err) {
      console.error(err);
      showToast(`Indexing failed: ${err.message}`, 'error');
      addMessage(`❌ Indexing error: ${err.message}`, false);
    } finally {
      setIsIndexing(false);
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
        <div style={{ marginTop: '18px' }}>
          <FileUpload onFilesAdded={handleFilesAdded} />
          
          <StatsCard uploadedFiles={uploadedFiles} />

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
import React, { useRef } from 'react';
import { generateId, formatBytes, getFileIcon } from '../utils/helpers';

const FileUpload = ({ onFilesAdded, uploadedFiles = [], showUploadedFiles = true }) => {
  const fileInputRef = useRef(null);

  const handleFiles = (files) => {
    if (files.length === 0) return;

    const newFiles = Array.from(files).map(f => {
      const id = generateId();
      const previewUrl = f.type.startsWith('image/') ? URL.createObjectURL(f) : null;
      return {
        id,
        file: f,
        name: f.name,
        type: f.type || 'application/octet-stream',
        size: f.size,
        status: 'pending',
        previewUrl
      };
    });

    onFilesAdded(newFiles);
  };

  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files || []);
    handleFiles(files);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <>
      <div className="upload-container">
        {/* Show uploaded files if any exist */}
        {uploadedFiles.length > 0 && (
          <div className="uploaded-files-section">
            <div className="uploaded-files-header">
              <span>📁 Uploaded Files ({uploadedFiles.length})</span>
            </div>
            <div className="uploaded-files-list">
              {uploadedFiles.map(file => (
                <div key={file.id} className="uploaded-file-item">
                  {file.previewUrl ? (
                    <div className="file-preview-small">
                      <img src={file.previewUrl} alt={file.name} />
                    </div>
                  ) : (
                    <div className="file-preview-small">
                      {getFileIcon(file.type)}
                    </div>
                  )}

                  <div className="file-info-small">
                    <div className="file-name-small" title={file.name}>{file.name}</div>
                    <div className="file-meta-small">
                      <span>{formatBytes(file.size)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload zone */}
        <div
          className="upload-zone"
          onClick={handleUploadClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <div className="upload-icon">
            {uploadedFiles.length > 0 ? '➕' : '☁️'}
          </div>
          <div className="upload-text">
            {uploadedFiles.length > 0
              ? 'Drop more files here or click to add more'
              : 'Drop files here or click to browse'
            }
          </div>
          <div className="upload-hint">
            Supports: PDF, TXT, DOCX, CSV, PNG, JPG, MP3, WAV, M4A
          </div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            multiple
            style={{ display: 'none' }}
          />
        </div>
      </div>
    </>
  );
};

export default FileUpload;

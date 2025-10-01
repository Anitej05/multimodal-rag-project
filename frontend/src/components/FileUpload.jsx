import React, { useRef } from 'react';
import { generateId } from '../utils/helpers';

const FileUpload = ({ onFilesAdded }) => {
  const fileInputRef = useRef(null);

  const handleFiles = (files) => {
    if (files.length === 0) return;
    
    const newFiles = files.map(f => {
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

  return (
    <>
      <div 
        className="upload-zone"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="upload-icon">☁️</div>
        <div className="upload-text">Drop files here or click to browse</div>
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
    </>
  );
};

export default FileUpload;
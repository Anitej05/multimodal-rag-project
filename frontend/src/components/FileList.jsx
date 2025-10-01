import React from 'react';
import { formatBytes, getFileIcon } from '../utils/helpers';

const FileList = ({ files }) => {
  if (files.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📁</div>
        <div className="empty-text">No files uploaded yet</div>
      </div>
    );
  }

  return (
    <div className="file-list">
      {files.map(file => (
        <div key={file.id} className="file-item">
          {file.previewUrl ? (
            <div className="file-preview">
              <img src={file.previewUrl} alt={file.name} />
            </div>
          ) : (
            <div className="file-preview">
              {getFileIcon(file.type)}
            </div>
          )}
          
          <div className="file-info">
            <div className="file-name">{file.name}</div>
            <div className="file-meta">
              <span>{formatBytes(file.size)}</span>
              <span>•</span>
              <span>{file.type.split('/')[1] || 'unknown'}</span>
            </div>
          </div>
          
          <div className={`file-status status-${file.status}`}>
            {file.status}
          </div>
        </div>
      ))}
    </div>
  );
};

export default FileList;
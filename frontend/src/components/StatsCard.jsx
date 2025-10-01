import React from 'react';
import { formatBytes } from '../utils/helpers';

const StatsCard = ({ uploadedFiles }) => {
  const totalFiles = uploadedFiles.length;
  const indexedFiles = uploadedFiles.filter(f => f.status === 'indexed').length;
  const totalSize = uploadedFiles.reduce((sum, f) => sum + f.size, 0);

  return (
    <div className="file-stats">
      <div className="stat-card">
        <span className="stat-value">{totalFiles}</span>
        <span className="stat-label">Total Files</span>
      </div>
      <div className="stat-card">
        <span className="stat-value">{indexedFiles}</span>
        <span className="stat-label">Indexed</span>
      </div>
      <div className="stat-card">
        <span className="stat-value">{formatBytes(totalSize)}</span>
        <span className="stat-label">Total Size</span>
      </div>
    </div>
  );
};

export default StatsCard;
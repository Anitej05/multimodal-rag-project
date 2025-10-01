import React from 'react';

const ConfirmDialog = ({ message, onConfirm, onCancel }) => {
  if (!message) return null;

  return (
    <div className="confirm-overlay">
      <div className="confirm-dialog">
        <div className="confirm-message">{message}</div>
        <div className="confirm-actions">
          <button className="btn btn-primary" onClick={onConfirm}>
            OK
          </button>
          <button className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
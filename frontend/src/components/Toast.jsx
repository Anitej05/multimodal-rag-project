import React from 'react';

const Toast = ({ message, type }) => {
  if (!message) return null;

  return (
    <div className={`toast toast-${type}`}>
      {message}
    </div>
  );
};

export default Toast;
import React, { useState } from 'react';
import Header from './components/Header';
import KnowledgeBase from './components/KnowledgeBase';
import Chat from './components/Chat';
import Toast from './components/Toast';
import { formatTime } from './utils/helpers';

const ThemeToggle = ({ isDark, onToggle }) => {
  return (
    <button
      onClick={onToggle}
      className="btn btn-secondary btn-icon"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000
      }}
    >
      {isDark ? '☀️' : '🌙'}
    </button>
  );
};

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [toast, setToast] = useState(null);

  // Initialize theme state from localStorage or default to light mode
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme ? JSON.parse(savedTheme) : false;
  });

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const addMessage = (text, isUser = false, sources = []) => {
    const time = formatTime();
    const newMessage = { text, isUser, time, sources };
    console.log('Adding message to state:', newMessage); // Debug
    setMessages(prev => [...prev, newMessage]);
  };

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem('theme', JSON.stringify(newTheme));
  };

  return (
    <div className={isDarkMode ? '' : 'light-theme'}>
      <ThemeToggle isDark={isDarkMode} onToggle={toggleTheme} />
      <Header />

      <main className="container">
        {/* Left Column - Knowledge Base */}
        <KnowledgeBase
          uploadedFiles={uploadedFiles}
          setUploadedFiles={setUploadedFiles}
          showToast={showToast}
          addMessage={addMessage}
        />

        {/* Right Column - Chat */}
        <Chat
          messages={messages}
          addMessage={addMessage}
          showToast={showToast}
        />
      </main>

      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  );
}

export default App;

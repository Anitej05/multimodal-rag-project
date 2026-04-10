import React, { useState } from 'react';
import Header from './components/Header';
import KnowledgeBase from './components/KnowledgeBase';
import Chat from './components/Chat';
import KnowledgeGraph from './components/KnowledgeGraph';
import Toast from './components/Toast';
import { formatTime } from './utils/helpers';
import './styles/knowledgeGraph.css';

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
  const [activeView, setActiveView] = useState('chat'); // 'chat' or 'graph'

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
    setMessages(prev => [...prev, newMessage]);
  };

  const updateLastMessage = (textOrUpdater) => {
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const updated = [...prev];
      const last = { ...updated[updated.length - 1] };
      if (typeof textOrUpdater === 'function') {
        textOrUpdater(last);
      } else {
        last.text = textOrUpdater;
      }
      updated[updated.length - 1] = last;
      return updated;
    });
  };

  const clearMessages = () => {
    setMessages([]);
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

      {/* View switcher tabs in the header area */}
      <div className="view-switcher" id="view-switcher">
        <button
          className={`view-tab ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveView('chat')}
          id="view-tab-chat"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          <span>Chat</span>
        </button>
        <button
          className={`view-tab ${activeView === 'graph' ? 'active' : ''}`}
          onClick={() => setActiveView('graph')}
          id="view-tab-graph"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <circle cx="12" cy="12" r="3"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="5" r="2"/>
            <circle cx="19" cy="19" r="2"/><circle cx="5" cy="19" r="2"/>
            <line x1="12" y1="9" x2="19" y2="7"/><line x1="12" y1="9" x2="5" y2="7"/>
            <line x1="12" y1="15" x2="19" y2="17"/><line x1="12" y1="15" x2="5" y2="17"/>
          </svg>
          <span>Knowledge Graph</span>
        </button>
      </div>

      {/* Chat view */}
      {activeView === 'chat' && (
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
            updateLastMessage={updateLastMessage}
            clearMessages={clearMessages}
            showToast={showToast}
          />
        </main>
      )}

      {/* Graph view */}
      {activeView === 'graph' && (
        <main className="container container-graph">
          {/* Left Column - Knowledge Base */}
          <KnowledgeBase
            uploadedFiles={uploadedFiles}
            setUploadedFiles={setUploadedFiles}
            showToast={showToast}
            addMessage={addMessage}
          />

          {/* Right Column - Knowledge Graph */}
          <KnowledgeGraph
            uploadedFiles={uploadedFiles}
            isVisible={activeView === 'graph'}
          />
        </main>
      )}

      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  );
}

export default App;

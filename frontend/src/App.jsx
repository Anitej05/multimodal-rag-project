import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import KnowledgeBase from './components/KnowledgeBase';
import Chat from './components/Chat';
import KnowledgeGraph from './components/KnowledgeGraph';
import Digitize from './components/Digitize';
import Toast from './components/Toast';
import api from './services/api';
import { formatTime } from './utils/helpers';
import './styles/knowledgeGraph.css';
import './styles/digitize.css';

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
  const [activeView, setActiveView] = useState('chat'); // 'chat', 'graph', or 'digitize'
  const [isIndexing, setIsIndexing] = useState(false);
  const [modeSwitching, setModeSwitching] = useState(false);

  // Mode switching handler — shows tab immediately, switches models in background
  const switchView = useCallback(async (view) => {
    if (view === activeView) return;
    const needsDigitize = view === 'digitize';
    const wasDigitize = activeView === 'digitize';

    setActiveView(view); // Switch tab immediately

    if (needsDigitize || wasDigitize) {
      setModeSwitching(true);
      try {
        await api.switchMode(needsDigitize ? 'digitize' : 'rag');
      } catch (err) {
        console.error('Mode switch failed:', err);
      }
      setModeSwitching(false);
    }
  }, [activeView]);

  // Resize state
  const [leftWidth, setLeftWidth] = useState(35); // percentage
  const [isResizing, setIsResizing] = useState(false);

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

  // Resizing logic
  const startResizing = useCallback(() => {
    setIsResizing(true);
  }, []);

  const stopResizing = useCallback(() => {
    setIsResizing(false);
  }, []);

  const resize = useCallback((e) => {
    if (isResizing) {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      if (newWidth > 20 && newWidth < 70) {
        setLeftWidth(newWidth);
      }
    }
  }, [isResizing]);

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', resize);
      window.addEventListener('mouseup', stopResizing);
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
    } else {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isResizing, resize, stopResizing]);

  return (
    <div className={isDarkMode ? '' : 'light-theme'}>
      <ThemeToggle isDark={isDarkMode} onToggle={toggleTheme} />
      <Header />

      {/* View switcher tabs in the header area */}
      <div className="view-switcher" id="view-switcher">
        <button
          className={`view-tab ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => switchView('chat')}
          disabled={modeSwitching}
          id="view-tab-chat"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          <span>Chat</span>
        </button>
        <button
          className={`view-tab ${activeView === 'graph' ? 'active' : ''}`}
          onClick={() => switchView('graph')}
          disabled={modeSwitching}
          id="view-tab-graph"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <circle cx="12" cy="12" r="3"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="5" r="2"/>
            <circle cx="19" cy="19" r="2"/><circle cx="5" cy="19" r="2"/>
            <line x1="12" y1="9" x2="19" y2="7"/><line x1="12" y1="9" x2="5" y2="7"/>
            <line x1="12" y1="15" x2="19" y2="17"/><line x1="12" y1="15" x2="5" y2="17"/>
          </svg>
          <span>Knowledge Graph</span>
          {isIndexing && <span className="kg-indexing-badge">●</span>}
        </button>
        <button
          className={`view-tab ${activeView === 'digitize' ? 'active' : ''}`}
          onClick={() => switchView('digitize')}
          disabled={modeSwitching}
          id="view-tab-digitize"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <rect x="4" y="2" width="16" height="20" rx="2"/>
            <path d="M8 8h8M8 12h8M8 16h5"/>
            <line x1="2" y1="18" x2="22" y2="6" strokeDasharray="2 2" opacity="0.5"/>
          </svg>
          <span>Digitize</span>
        </button>
      </div>

      {/* Main content */}
      {activeView === 'digitize' ? (
        /* Digitize gets full-width layout outside the grid */
        <main className="container container-digitize" style={{ display: 'flex' }}>
          <Digitize showToast={showToast} setUploadedFiles={setUploadedFiles} />
        </main>
      ) : (
        <main 
          className={`container ${activeView === 'graph' ? 'container-graph' : ''} ${isResizing ? 'resizing' : ''}`}
          style={{ gridTemplateColumns: `${leftWidth}% 4px minmax(0, 1fr)` }}
        >
          {/* Left Column - Knowledge Base */}
          <KnowledgeBase
            uploadedFiles={uploadedFiles}
            setUploadedFiles={setUploadedFiles}
            showToast={showToast}
            addMessage={addMessage}
            onIndexingChange={setIsIndexing}
          />

          {/* Resizer Handle */}
          <div 
            className={`resizer ${isResizing ? 'active' : ''}`} 
            onMouseDown={startResizing}
            title="Drag to resize"
          />

          {/* Right Column */}
          <div style={{ display: activeView === 'chat' ? 'flex' : 'none', flexDirection: 'column', minHeight: 0 }}>
            <Chat
              messages={messages}
              addMessage={addMessage}
              updateLastMessage={updateLastMessage}
              clearMessages={clearMessages}
              showToast={showToast}
            />
          </div>

          {activeView === 'graph' && (
            <KnowledgeGraph
              uploadedFiles={uploadedFiles}
              isVisible={true}
              isIndexing={isIndexing}
            />
          )}
        </main>
      )}

      {/* Mode switching overlay */}
      {modeSwitching && (
        <div className="dg-mode-overlay">
          <div className="dg-mode-spinner" />
          <h3>Switching Mode...</h3>
          <p>Managing GPU memory — unloading and loading models for optimal performance.</p>
        </div>
      )}

      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  );
}

export default App;

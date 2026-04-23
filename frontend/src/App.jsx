import React, { useState, useCallback, useRef, useEffect } from 'react';
import Header from './components/Header';
import KnowledgeBase from './components/KnowledgeBase';
import Chat from './components/Chat';
import KnowledgeGraph from './components/KnowledgeGraph';
import Digitize from './components/Digitize';
import Toast from './components/Toast';
import { formatTime } from './utils/helpers';
import api from './services/api';
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
  const [gpuMode, setGpuMode] = useState('rag'); // 'rag' or 'digitize'
  const [isSwitchingMode, setIsSwitchingMode] = useState(false);
  const switchingRef = useRef(false);
  const [splitRatio, setSplitRatio] = useState(36); // KB panel width %
  const isDragging = useRef(false);
  const containerRef = useRef(null);

  // ── Draggable splitter handlers ──
  const handleSplitterMouseDown = useCallback((e) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = (x / rect.width) * 100;
      setSplitRatio(Math.min(60, Math.max(20, pct))); // clamp 20-60%
    };
    const handleMouseUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleViewChange = useCallback(async (newView) => {
    if (newView === activeView) return;
    const needsDigitize = newView === 'digitize';
    const needsRag = newView !== 'digitize';
    const currentMode = gpuMode;

    // Only switch GPU mode if actually changing between digitize and non-digitize
    if (needsDigitize && currentMode !== 'digitize') {
      if (switchingRef.current) return;
      switchingRef.current = true;
      setIsSwitchingMode(true);
      try {
        await api.switchMode('digitize');
        setGpuMode('digitize');
      } catch (e) {
        console.warn('Mode switch to digitize failed:', e);
      }
      setIsSwitchingMode(false);
      switchingRef.current = false;
    } else if (needsRag && currentMode !== 'rag') {
      if (switchingRef.current) return;
      switchingRef.current = true;
      setIsSwitchingMode(true);
      try {
        await api.switchMode('rag');
        setGpuMode('rag');
      } catch (e) {
        console.warn('Mode switch to rag failed:', e);
      }
      setIsSwitchingMode(false);
      switchingRef.current = false;
    }

    setActiveView(newView);
  }, [activeView, gpuMode]);

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
          onClick={() => handleViewChange('chat')}
          id="view-tab-chat"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
          </svg>
          <span>Chat</span>
        </button>
        <button
          className={`view-tab ${activeView === 'graph' ? 'active' : ''}`}
          onClick={() => handleViewChange('graph')}
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
          onClick={() => handleViewChange('digitize')}
          id="view-tab-digitize"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <path d="M3 7V5a2 2 0 012-2h2m10 0h2a2 2 0 012 2v2m0 10v2a2 2 0 01-2 2h-2M3 17v2a2 2 0 002 2h2"/>
            <line x1="7" y1="12" x2="17" y2="12"/>
          </svg>
          <span>Digitize</span>
        </button>
      </div>

      {/* Main content */}
      <main
        ref={containerRef}
        className={`container ${activeView === 'graph' ? 'container-graph' : ''} ${activeView === 'digitize' ? 'container-digitize' : ''}`}
        style={activeView === 'chat' ? { gridTemplateColumns: `${splitRatio}% 6px 1fr` } : undefined}
      >
        {/* Left Column - Knowledge Base (hidden in digitize mode) */}
        <div style={{ display: activeView === 'digitize' ? 'none' : 'contents' }}>
          <KnowledgeBase
            uploadedFiles={uploadedFiles}
            setUploadedFiles={setUploadedFiles}
            showToast={showToast}
            addMessage={addMessage}
            onIndexingChange={setIsIndexing}
          />
        </div>

        {/* Draggable Splitter — only in chat view */}
        {activeView === 'chat' && (
          <div
            className="panel-splitter"
            onMouseDown={handleSplitterMouseDown}
            title="Drag to resize"
          >
            <div className="panel-splitter-line" />
          </div>
        )}

        {/* Right Column - Chat */}
        <div style={{ display: activeView === 'chat' ? 'contents' : 'none' }}>
          <Chat
            messages={messages}
            addMessage={addMessage}
            updateLastMessage={updateLastMessage}
            clearMessages={clearMessages}
            showToast={showToast}
          />
        </div>

        {activeView === 'digitize' && (
          <Digitize showToast={showToast} setUploadedFiles={setUploadedFiles} />
        )}

        {activeView === 'graph' && (
          <KnowledgeGraph
            uploadedFiles={uploadedFiles}
            isVisible={true}
            isIndexing={isIndexing}
          />
        )}
      </main>

      {/* Mode Switching Overlay */}
      {isSwitchingMode && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(6px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 9999, flexDirection: 'column', gap: '16px'
        }}>
          <div style={{
            width: '40px', height: '40px', border: '3px solid rgba(255,255,255,0.2)',
            borderTopColor: '#a78bfa', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite'
          }} />
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 600 }}>
            {gpuMode === 'rag' ? 'Loading OCR engine on GPU...' : 'Reloading RAG models on GPU...'}
          </div>
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>Switching GPU mode</div>
        </div>
      )}

      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  );
}

export default App;

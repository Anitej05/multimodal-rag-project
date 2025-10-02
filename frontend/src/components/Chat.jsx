import React, { useState, useRef, useEffect } from 'react';
import api from '../services/api';
import ConfirmDialog from './ConfirmDialog';

const Chat = ({ messages, addMessage, showToast }) => {
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [recorder, setRecorder] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const chatMessagesRef = useRef(null);
  const chatInputRef = useRef(null);
  const audioInputRef = useRef(null);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const sendMessage = async () => {
    const text = chatInput.trim();
    const audio = audioFile;

    if (!text && !audio) {
      showToast('Please enter a message or attach an audio file', 'error');
      return;
    }

    if (audio) {
      addMessage('🎤 [Audio message]', true, []);
      setChatInput('');
      setAudioFile(null);
      if (audioInputRef.current) audioInputRef.current.value = '';

      setIsTyping(true);
      try {
        const res = await api.chatAudio(audio);
        const messageText = res.answer || res.response || 'No response from assistant';
        let sources = [];
        if (res.source) {
          sources = [{ source: res.source }];
        } else if (res.sources) {
          sources = Array.isArray(res.sources) ? res.sources : [];
        } else if (res.source_documents) {
          sources = Array.isArray(res.source_documents) ? res.source_documents : [];
        }
        addMessage(messageText, false, sources);
      } catch (err) {
        console.error(err);
        addMessage(`❌ Audio chat error: ${err.message}`, false, []);
        showToast(`Audio chat failed: ${err.message}`, 'error');
      } finally {
        setIsTyping(false);
      }
      return;
    }

    addMessage(text, true, []);
    setChatInput('');
    if (chatInputRef.current) {
      chatInputRef.current.style.height = 'auto';
    }
    
    setIsTyping(true);
    try {
      const res = await api.chatText(text);
      const messageText = res.answer || res.response || 'No response from assistant';
      let sources = [];
      if (res.source) {
        sources = [{ source: res.source }];
      } else if (res.sources) {
        sources = Array.isArray(res.sources) ? res.sources : [];
      } else if (res.source_documents) {
        sources = Array.isArray(res.source_documents) ? res.source_documents : [];
      }
      addMessage(messageText, false, sources);
    } catch (err) {
      console.error(err);
      addMessage(`❌ Error: ${err.message}`, false, []);
      showToast(`Chat failed: ${err.message}`, 'error');
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleChatInputChange = (e) => {
    setChatInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  const handleAudioInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
      showToast('Audio file attached. Click Send to submit.', 'info');
    }
  };

  const toggleRecording = async () => {
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
      setIsRecording(false);
      showToast('Recording stopped', 'info');
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Recording not supported in this browser', 'error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size) chunks.push(ev.data);
      };
      
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const file = new File([blob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
        setAudioFile(file);
        showToast('Recording ready! Click Send to submit.', 'success');
        setRecorder(null);
        setIsRecording(false);
      };
      
      mediaRecorder.start();
      setRecorder(mediaRecorder);
      setIsRecording(true);
      showToast('Recording started...', 'info');
    } catch (err) {
      console.error(err);
      showToast(`Microphone access denied: ${err.message}`, 'error');
    }
  };

  const clearChat = () => {
    setShowConfirm(true);
  };

  const handleConfirmClear = () => {
    setShowConfirm(false);
    showToast('Chat cleared', 'info');
    window.location.reload();
  };

  const handleCancelClear = () => {
    setShowConfirm(false);
  };

  return (
    <>
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-icon">💬</div>
            AI Assistant
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages" ref={chatMessagesRef}>
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🤖</div>
                <div className="empty-text">Start a conversation</div>
                <div style={{ fontSize: '13px', marginTop: '4px' }}>
                  Ask questions about your uploaded documents, images, or audio files
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, idx) => {
                  // Check if sources exist and are valid
                  const hasSources = !msg.isUser && 
                                    msg.sources && 
                                    Array.isArray(msg.sources) && 
                                    msg.sources.length > 0;
                  
                  return (
                  <div key={idx} className={`message ${msg.isUser ? 'message-user' : 'message-assistant'}`}>
                    <div className="message-avatar">
                      {msg.isUser ? '👤' : '🤖'}
                    </div>
                    <div className="message-content">
                      <div className="message-bubble">
                        {msg.text.split('\n').map((line, i) => (
                          <React.Fragment key={i}>
                            {line}
                            {i < msg.text.split('\n').length - 1 && <br />}
                          </React.Fragment>
                        ))}
                      </div>
                      {hasSources && (
                        <div className="message-sources">
                          <div className="sources-header">📚 Sources:</div>
                          {msg.sources.map((source, sIdx) => {
                            let filename = 'Unknown Source';
                            
                            if (typeof source === 'string') {
                              filename = source;
                            } else if (source && typeof source === 'object') {
                              filename = source.source || 
                                        source.metadata?.filename || 
                                        source.metadata?.source || 
                                        source.filename ||
                                        `Document ${sIdx + 1}`;
                            }
                            
                            if (filename && (filename.includes('/') || filename.includes('\\'))) {
                              filename = filename.split(/[/\\]/).pop();
                            }
                            
                            return (
                              <div key={sIdx} className="source-item">
                                <span className="source-icon">📄</span>
                                <span className="source-name">{filename}</span>
                                {source && source.metadata && source.metadata.page && (
                                  <span className="source-page">Page {source.metadata.page}</span>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                      <div className="message-time">{msg.time}</div>
                    </div>
                  </div>
                )})}
                {isTyping && (
                  <div className="message message-assistant">
                    <div className="message-avatar">🤖</div>
                    <div className="message-content">
                      <div className="message-bubble">
                        <span className="loading-spinner"></span>
                        <span style={{ marginLeft: '8px' }}>Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="chat-input-container">
            <div className="chat-input-wrapper">
              <textarea 
                ref={chatInputRef}
                className="chat-input"
                placeholder="Ask anything about your knowledge base..."
                rows="1"
                value={chatInput}
                onChange={handleChatInputChange}
                onKeyPress={handleKeyPress}
              />
              <button 
                className="btn btn-primary btn-icon" 
                onClick={sendMessage}
                title="Send message"
              >
                <span>📤</span>
              </button>
            </div>
            
            <div className="input-actions">
              <label className="btn btn-secondary btn-icon" title="Attach audio file">
                <input 
                  ref={audioInputRef}
                  type="file" 
                  accept="audio/*" 
                  onChange={handleAudioInputChange}
                  style={{ display: 'none' }}
                />
                <span>🎤</span>
              </label>
              <button 
                className={`btn btn-secondary btn-icon ${isRecording ? 'recording' : ''}`}
                onClick={toggleRecording}
                title="Record audio"
              >
                <span>{isRecording ? '⏹️' : '⏺️'}</span>
              </button>
              <button 
                className="btn btn-secondary btn-icon"
                onClick={clearChat}
                title="Clear chat"
              >
                <span>🗑️</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <ConfirmDialog 
        message={showConfirm ? "Clear all chat messages?" : null}
        onConfirm={handleConfirmClear}
        onCancel={handleCancelClear}
      />
    </>
  );
};

export default Chat;
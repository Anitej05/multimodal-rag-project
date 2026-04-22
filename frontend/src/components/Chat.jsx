import React, { useState, useRef, useEffect, useCallback } from 'react';
import api from '../services/api';
import ConfirmDialog from './ConfirmDialog';

const Chat = ({ messages, addMessage, updateLastMessage, clearMessages, showToast }) => {
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [recorder, setRecorder] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [availableFiles, setAvailableFiles] = useState([]);

  const chatMessagesRef = useRef(null);
  const chatInputRef = useRef(null);
  const streamBubbleRef = useRef(null);
  const streamContentRef = useRef(''); // Accumulate tokens without re-render
  const rafRef = useRef(null);
  const userScrolledUpRef = useRef(false); // Track if user manually scrolled up

  // Fetch list of available files so we can match source names to actual filenames
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const result = await api.listFiles();
        if (result.files) {
          setAvailableFiles(result.files);
        }
      } catch (err) {
        console.log('Could not fetch file list:', err);
      }
    };
    fetchFiles();
  }, [messages]);

  // Check if user is near the bottom of chat (within 80px threshold)
  const isNearBottom = useCallback(() => {
    const el = chatMessagesRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  // Track user scroll to detect manual scroll-up
  const handleChatScroll = useCallback(() => {
    userScrolledUpRef.current = !isNearBottom();
  }, [isNearBottom]);

  useEffect(() => {
    if (chatMessagesRef.current && !userScrolledUpRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, isTyping, isStreaming]);

  // Auto-scroll during streaming via RAF — only if user hasn't scrolled up
  const scrollToBottom = useCallback(() => {
    if (chatMessagesRef.current && !userScrolledUpRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, []);

  // Match a clean source name (without extension) to an actual filename  
  const findActualFilename = useCallback((cleanName) => {
    if (!cleanName || !availableFiles.length) return null;

    // First try exact match (with extension)
    const exactMatch = availableFiles.find(f => f.name === cleanName);
    if (exactMatch) return exactMatch.name;

    // Try matching without extension
    const normalizedClean = cleanName.replace(/[_\s]+/g, ' ').toLowerCase().trim();
    for (const file of availableFiles) {
      const nameWithoutExt = file.name.replace(/\.[^/.]+$/, '');
      const normalizedFile = nameWithoutExt.replace(/[_\s]+/g, ' ').toLowerCase().trim();
      if (normalizedFile === normalizedClean) {
        return file.name;
      }
    }

    // Try partial match
    for (const file of availableFiles) {
      const nameWithoutExt = file.name.replace(/\.[^/.]+$/, '');
      const normalizedFile = nameWithoutExt.replace(/[_\s]+/g, ' ').toLowerCase().trim();
      if (normalizedFile.includes(normalizedClean) || normalizedClean.includes(normalizedFile)) {
        return file.name;
      }
    }

    return null;
  }, [availableFiles]);

  // Handle click events on source items using event delegation
  const handleSourceClick = useCallback((e) => {
    const sourceItem = e.target.closest('.source-item');
    if (!sourceItem) return;

    // Prefer the data-filename attribute set by backend (exact filename with extension)
    let actualFilename = sourceItem.getAttribute('data-filename');

    // Fallback: try fuzzy matching from available files
    if (!actualFilename) {
      const sourceName = sourceItem.querySelector('.source-name');
      if (sourceName) {
        actualFilename = findActualFilename(sourceName.textContent.trim());
      }
    }

    if (actualFilename) {
      const fileUrl = api.getFilePreviewUrl(actualFilename);
      window.open(fileUrl, '_blank');
    } else {
      showToast('Could not open file', 'error');
    }
  }, [findActualFilename, showToast]);

  // Attach click handler to source items via event delegation
  useEffect(() => {
    const container = chatMessagesRef.current;
    if (!container) return;

    container.addEventListener('click', handleSourceClick);
    return () => container.removeEventListener('click', handleSourceClick);
  }, [handleSourceClick]);

  const sendMessage = async () => {
    const text = chatInput.trim();

    if (!text) {
      showToast('Please enter a message', 'error');
      return;
    }

    addMessage(text, true, {});
    setChatInput('');
    
    setIsTyping(true);
    setIsStreaming(false);
    streamContentRef.current = '';
    userScrolledUpRef.current = false; // Reset scroll lock when sending new message

    try {
      let sourcesData = null;

      await api.chatTextStream(text, {
        onToken: (token) => {
          // First token received — switch from "searching" to streaming
          if (!streamContentRef.current) {
            setIsTyping(false);
            setIsStreaming(true);
          }

          streamContentRef.current += token;

          // Use requestAnimationFrame for smooth DOM updates
          if (rafRef.current) cancelAnimationFrame(rafRef.current);
          rafRef.current = requestAnimationFrame(() => {
            if (streamBubbleRef.current) {
              // Only strip code fences — leave everything else as the LLM sends it
              let content = streamContentRef.current;
              content = content.replace(/```(?:html|markdown|text|md|)\s*\n?/gi, '');
              streamBubbleRef.current.innerHTML = content;
              scrollToBottom();
            }
          });
        },
        onSources: (data) => {
          sourcesData = data;
        },
        onDone: () => {
          // Cancel any pending RAF
          if (rafRef.current) cancelAnimationFrame(rafRef.current);

          // Build final message with sources
          let finalContent = streamContentRef.current;
          // Only strip code fences
          finalContent = finalContent.replace(/```(?:html|markdown|text|md|)\s*\n?/gi, '');

          
          if (sourcesData && sourcesData.length > 0) {
            // Build sources HTML
            let sourcesHtml = '<div class="sources-section"><h3>Sources</h3>';
            sourcesData.forEach(src => {
              sourcesHtml += `<div class="source-item source-item-clickable" data-filename="${src.filename}"><span class="source-key">${src.key}</span><span class="source-name">${src.name}</span><span class="source-open-icon" title="Click to open file">↗</span></div>`;
            });
            sourcesHtml += '</div>';
            finalContent = `${finalContent}--%Sources%--${sourcesHtml}`;
          }

          // Commit to React state
          addMessage(finalContent, false, {});
          setIsStreaming(false);
          setIsTyping(false);
          streamContentRef.current = '';
        },
        onError: (err) => {
          console.error('Stream error:', err);
          if (rafRef.current) cancelAnimationFrame(rafRef.current);

          const errorContent = streamContentRef.current || `❌ Error: ${err}`;
          addMessage(errorContent, false, {});
          setIsStreaming(false);
          setIsTyping(false);
          streamContentRef.current = '';
        }
      });
    } catch (err) {
      console.error(err);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      
      if (!streamContentRef.current) {
        addMessage(`❌ Error: ${err.message}`, false, {});
      }
      showToast(`Chat failed: ${err.message}`, 'error');
      setIsStreaming(false);
      setIsTyping(false);
      streamContentRef.current = '';
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
      
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const file = new File([blob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
        
        try {
          showToast('Transcribing audio...', 'info');
          const result = await api.transcribeAudio(file);
          if (result.transcription) {
            setChatInput(result.transcription);
            showToast('Audio transcribed! You can review and edit before sending.', 'success');
          } else {
            showToast('Could not understand the audio. Please try again.', 'error');
          }
        } catch (err) {
          console.error('Transcription error:', err);
          showToast(`Transcription failed: ${err.message}`, 'error');
        }
        
        setRecorder(null);
        setIsRecording(false);
      };
      
      mediaRecorder.start();
      setRecorder(mediaRecorder);
      setIsRecording(true);
      showToast('Recording started... Speak now', 'info');
    } catch (err) {
      console.error(err);
      showToast(`Microphone access denied: ${err.message}`, 'error');
    }
  };

  const readAloud = async () => {
    // Use browser's built-in SpeechSynthesis — works offline, no server calls
    if (!('speechSynthesis' in window)) {
      showToast('Text-to-speech not supported in this browser', 'error');
      return;
    }

    // If already speaking, toggle pause/resume
    if (window.speechSynthesis.speaking) {
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        showToast('Resumed playback', 'info');
      } else {
        window.speechSynthesis.pause();
        showToast('Paused playback', 'info');
      }
      return;
    }

    const lastAIMessage = [...messages].reverse().find(msg => !msg.isUser);
    if (!lastAIMessage || !lastAIMessage.text) {
      showToast('No AI response to read aloud', 'error');
      return;
    }

    // Strip HTML tags and source markers for clean speech
    const cleanText = lastAIMessage.text
      .split('--%Sources%--')[0]
      .replace(/<[^>]*>/g, '')
      .replace(/\s+/g, ' ')
      .trim();

    if (!cleanText) {
      showToast('No text content to read', 'error');
      return;
    }

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Prefer a female / natural-sounding voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice =
      voices.find(v => v.lang.startsWith('en') && /zira|jenny|aria|female|woman|samantha|karen|fiona|hazel|susan|linda/i.test(v.name)) ||
      voices.find(v => v.lang.startsWith('en') && /Google.*Female|Microsoft.*Online/i.test(v.name)) ||
      voices.find(v => v.lang.startsWith('en')) ||
      voices[0];
    if (preferredVoice) utterance.voice = preferredVoice;

    utterance.onend = () => showToast('Finished reading', 'info');
    utterance.onerror = () => showToast('Speech synthesis error', 'error');

    window.speechSynthesis.cancel(); // Cancel any pending
    window.speechSynthesis.speak(utterance);
    showToast('Reading aloud...', 'info');
  };

  const clearChat = () => {
    setShowConfirm(true);
  };

  const handleConfirmClear = () => {
    setShowConfirm(false);
    clearMessages();
    showToast('Chat cleared', 'info');
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
          <div className="chat-messages" ref={chatMessagesRef} onScroll={handleChatScroll}>
            {messages.length === 0 && !isStreaming ? (
              <div className="empty-state">
                <div className="empty-icon">🤖</div>
                <div className="empty-text">Start a conversation</div>
                <div style={{ fontSize: '13px', marginTop: '4px' }}>
                  Ask questions about your uploaded documents or images
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, idx) => {
                  // Split the message text by the sources separator
                  const sourceMarker = '--%Sources%--';
                  const parts = msg.text.split(sourceMarker);
                  const contentText = parts[0] || msg.text;
                  let sourcesHtml = parts[1] || '';

                  // Extract document names from sources HTML for tooltip mapping
                  const documentMap = {};
                  if (sourcesHtml) {
                    // Match both <p> and <span> source-name patterns
                    const sourceItemRegex = /<div class="source-item[^"]*"[^>]*>.*?<(?:p|span) class="source-name">([^<]+)<\/(?:p|span)>.*?<\/div>/gs;
                    let match;
                    let index = 0;
                    while ((match = sourceItemRegex.exec(sourcesHtml)) !== null) {
                      index++;
                      documentMap[index.toString()] = match[1];
                    }
                  }

                  // Process the content to wrap plain bracketed citations with styled spans
                  let processedContentText = contentText;

                  processedContentText = processedContentText.replace(/\[([0-9]+)\]/g, function(match, number) {
                    const documentName = documentMap[number] || `Document ${number}`;
                    return `<span class="citation" data-source="${number}" data-source-name="${documentName}">${number}</span>`;
                  });

                  // Process sources HTML to remove brackets and add clickable styling
                  if (sourcesHtml) {
                    const keyRegex = new RegExp('<span class="source-key">\\[([0-9]+)\\]<\\/span>', 'g');
                    sourcesHtml = sourcesHtml.replace(keyRegex, '<span class="source-key">$1</span>');

                    // Add clickable class to source items (preserve any existing attributes like data-filename)
                    sourcesHtml = sourcesHtml.replace(
                      /<div class="source-item"([^>]*)>/g,
                      '<div class="source-item source-item-clickable"$1>'
                    );

                    // Add the open icon after each source-name (only if not already present)
                    if (!sourcesHtml.includes('source-open-icon')) {
                      sourcesHtml = sourcesHtml.replace(
                        /(<(?:p|span) class="source-name">([^<]+)<\/(?:p|span)>)/g,
                        '$1<span class="source-open-icon" title="Click to open file">↗</span>'
                      );
                    }
                  }

                  // Cleanup
                  processedContentText = processedContentText.replace(/--%Sources%--/g, '');
                  if (sourcesHtml) {
                    sourcesHtml = sourcesHtml.replace(/--%Sources%--/g, '');
                  }

                  return (
                    <div key={idx} className={`message ${msg.isUser ? 'message-user' : 'message-assistant'}`}>
                      <div className="message-avatar">
                        {msg.isUser ? '👤' : '🤖'}
                      </div>
                      <div className="message-content">
                        <div className="message-bubble" dangerouslySetInnerHTML={{ __html: processedContentText }}></div>
                        {sourcesHtml && (
                          <div className="message-sources" dangerouslySetInnerHTML={{ __html: sourcesHtml }}></div>
                        )}
                        <div className="message-time">{msg.time}</div>
                      </div>
                    </div>
                  );
                })}

                {/* Live streaming bubble - renders outside React state for performance */}
                {isStreaming && (
                  <div className="message message-assistant">
                    <div className="message-avatar">🤖</div>
                    <div className="message-content">
                      <div className="message-bubble streaming" ref={streamBubbleRef}></div>
                    </div>
                  </div>
                )}

                {/* Searching indicator - shown before first token arrives */}
                {isTyping && (
                  <div className="message message-assistant">
                    <div className="message-avatar">🤖</div>
                    <div className="message-content">
                      <div className="message-bubble">
                        <span className="loading-spinner"></span>
                        <span style={{ marginLeft: '8px' }}>Searching knowledge base...</span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="chat-input-container">
            <div className="chat-input-wrapper">
              <input 
                ref={chatInputRef}
                type="text"
                className="chat-input"
                placeholder="Ask anything about your knowledge base..."
                value={chatInput}
                onChange={handleChatInputChange}
                onKeyPress={handleKeyPress}
              />
              <button 
                className="btn btn-primary btn-icon" 
                onClick={sendMessage}
                title="Send message"
                disabled={isTyping || isStreaming}
              >
                <span>📤</span>
              </button>
            </div>
            
            <div className="input-actions">
              <button 
                className="btn btn-secondary btn-icon"
                onClick={readAloud}
                title="Read aloud"
              >
                <span>🔊</span>
              </button>
              <button 
                className={`btn btn-secondary btn-icon ${isRecording ? 'recording' : ''}`}
                onClick={toggleRecording}
                title="Record audio"
              >
                <span>{isRecording ? '🟥' : '🎙️'}</span>
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

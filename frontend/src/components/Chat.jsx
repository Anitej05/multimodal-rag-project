import React, { useState, useRef, useEffect } from 'react';
import api from '../services/api';
import ConfirmDialog from './ConfirmDialog';

const Chat = ({ messages, addMessage, showToast }) => {
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [recorder, setRecorder] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [audio, setAudio] = useState(null);

  const chatMessagesRef = useRef(null);
  const chatInputRef = useRef(null);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const sendMessage = async () => {
    const text = chatInput.trim();

    if (!text) {
      showToast('Please enter a message', 'error');
      return;
    }

    addMessage(text, true, {});
    setChatInput('');
    if (chatInputRef.current) {
      chatInputRef.current.style.height = 'auto';
    }
    
    setIsTyping(true);
    try {
      const res = await api.chatText(text);
      const messageText = res.answer || res.response || 'No response from assistant';
      addMessage(messageText, false, {});
    } catch (err) {
      console.error(err);
      addMessage(`❌ Error: ${err.message}`, false, {});
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
        
        // Transcribe the audio and put it in the input box
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
    if (audio && !audio.paused) {
      audio.pause();
      showToast('Audio paused', 'info');
      return;
    }

    if (audio && audio.paused) {
      audio.play();
      showToast('Playing audio...', 'info');
      return;
    }

    const lastAIMessage = [...messages].reverse().find(msg => !msg.isUser);
    if (!lastAIMessage || !lastAIMessage.text) {
      showToast('No AI response to read aloud', 'error');
      return;
    }

    try {
      showToast('Generating audio...', 'info');
      const audioBlob = await api.generateAudio(lastAIMessage.text);
      const audioUrl = URL.createObjectURL(audioBlob);
      const newAudio = new Audio(audioUrl);
      
      newAudio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setAudio(null);
        console.log('Audio playback finished, URL revoked');
      };

      newAudio.onerror = () => {
        URL.revokeObjectURL(audioUrl);
        setAudio(null);
        showToast('Error playing audio', 'error');
        console.log('Audio error occurred, URL revoked');
      };

      setAudio(newAudio);
      newAudio.play();
      showToast('Playing audio...', 'info');
    } catch (err) {
      console.error('Error generating audio:', err);
      showToast(`Audio generation failed: ${err.message}`, 'error');
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
                    // Parse the sources HTML to map citation numbers to document names
                    // The sources follow a pattern where we need to extract filename from each source-item
                    const sourceItemRegex = /<div class="source-item"><p class="source-name">([^<]+)<\/p><\/div>/g;
                    const sourceItems = sourcesHtml.match(sourceItemRegex) || [];

                    // Assign document names in order of citation numbers
                    sourceItems.forEach((item, index) => {
                      const citationNumber = (index + 1).toString();
                      const nameMatch = item.match(/<p class="source-name">([^<]+)<\/p>/);
                      if (nameMatch && nameMatch[1]) {
                        documentMap[citationNumber] = nameMatch[1];
                      }
                    });
                  }

                  // Process the content to wrap plain bracketed citations with styled spans
                  // Replace [number] with styled span showing only the number
                  let processedContentText = contentText;

                  // Debug: Log the documentMap to see what's being extracted
                  console.log('DEBUG: documentMap:', documentMap);
                  console.log('DEBUG: contentText before processing:', contentText);

                  processedContentText = processedContentText.replace(/\[([0-9]+)\]/g, function(match, number) {
                    const documentName = documentMap[number] || `Document ${number}`;
                    console.log(`DEBUG: Citation [${number}] -> Document name: "${documentName}"`);
                    return `<span class="citation" data-source="${number}" data-source-name="${documentName}">${number}</span>`;
                  });

                  console.log('DEBUG: processedContentText after:', processedContentText);

                  // Process sources HTML to remove brackets from citation numbers
                  if (sourcesHtml) {
                    // Remove brackets from source-key elements that contain [number] format
                    const keyRegex = new RegExp('<span class="source-key">\\[([0-9]+)\\]</span>', 'g');
                    sourcesHtml = sourcesHtml.replace(keyRegex, '<span class="source-key">$1</span>');
                  }

                  // Comprehensive cleanup: Remove any remaining source markers that might have slipped through
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

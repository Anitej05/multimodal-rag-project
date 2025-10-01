import React, { useState } from 'react';
import Header from './components/Header';
import KnowledgeBase from './components/KnowledgeBase';
import Chat from './components/Chat';
import Toast from './components/Toast';
import { formatTime } from './utils/helpers';

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const addMessage = (text, isUser = false) => {
    const time = formatTime();
    setMessages(prev => [...prev, { text, isUser, time }]);
  };

  return (
    <>
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
    </>
  );
}

export default App;
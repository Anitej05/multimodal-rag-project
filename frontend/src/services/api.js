const API_BASE_URL = 'http://127.0.0.1:8000';

const api = {
  async ingestFiles(files) {
    const fd = new FormData();
    for (const f of files) {
      fd.append('files', f);
    }
    fd.append('reset_db', 'false');
    const res = await fetch(`${API_BASE_URL}/ingest`, { 
      method: 'POST', 
      body: fd 
    });
    if (!res.ok) throw new Error(`Ingest failed: ${res.statusText}`);
    return res.json();
  },

  async getIngestStatus() {
    const res = await fetch(`${API_BASE_URL}/ingest-status`);
    if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
    return res.json();
  },

  async getKnowledgeGraph() {
    const res = await fetch(`${API_BASE_URL}/knowledge-graph`);
    if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`);
    return res.json();
  },

  async chatText(query) {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);
    return res.json();
  },

  async chatTextStream(query, { onToken, onSources, onDone, onError }) {
    try {
      const res = await fetch(`${API_BASE_URL}/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (!res.ok) throw new Error(`Chat stream failed: ${res.statusText}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          const data = trimmed.slice(6);

          if (data === '[DONE]') {
            if (onDone) onDone();
            return;
          }

          try {
            const parsed = JSON.parse(data);
            if (parsed.token && onToken) {
              onToken(parsed.token);
            }
            if (parsed.sources && onSources) {
              onSources(parsed.sources);
            }
            if (parsed.error && onError) {
              onError(parsed.error);
            }
          } catch (e) {
            // Skip malformed JSON chunks
          }
        }
      }
      if (onDone) onDone();
    } catch (err) {
      if (onError) onError(err.message);
      throw err;
    }
  },

  async chatAudio(file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API_BASE_URL}/chat-audio`, { 
      method: 'POST', 
      body: fd 
    });
    if (!res.ok) throw new Error(`Chat audio failed: ${res.statusText}`);
    return res.json();
  },

   async transcribeAudio(file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API_BASE_URL}/transcribe`, { 
      method: 'POST', 
      body: fd 
    });
    if (!res.ok) throw new Error(`Transcription failed: ${res.statusText}`);
    return res.json();
  },

  async generateAudio(text, voice = 'af_heart') {
    const fd = new FormData();
    fd.append('text', text);
    fd.append('voice', voice);
    const res = await fetch(`${API_BASE_URL}/generate_audio`, { 
      method: 'POST', 
      body: fd 
    });
    if (!res.ok) throw new Error(`Audio generation failed: ${res.statusText}`);
    return res.blob();
  },

  async resetKB() {
    const res = await fetch(`${API_BASE_URL}/reset`, { 
      method: 'POST' 
    });
    if (!res.ok) throw new Error(`Reset failed: ${res.statusText}`);
    return res.json();
  },

  getFileUrl(filename) {
    return `${API_BASE_URL}/files/${encodeURIComponent(filename)}`;
  },

  getFilePreviewUrl(filename) {
    return `${API_BASE_URL}/files/${encodeURIComponent(filename)}/preview`;
  },

  async listFiles() {
    const res = await fetch(`${API_BASE_URL}/files`);
    if (!res.ok) throw new Error(`List files failed: ${res.statusText}`);
    return res.json();
  },

  // --- Mode Switching ---
  async switchMode(mode) {
    const res = await fetch(`${API_BASE_URL}/switch-mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    if (!res.ok) throw new Error(`Switch mode failed: ${res.statusText}`);
    return res.json();
  },

  async getModeStatus() {
    const res = await fetch(`${API_BASE_URL}/mode-status`);
    if (!res.ok) throw new Error(`Mode status failed: ${res.statusText}`);
    return res.json();
  },

  // --- Digitize ---
  async digitize(files, lang = 'en') {
    const fd = new FormData();
    for (const f of files) {
      fd.append('files', f);
    }
    fd.append('lang', lang);
    const res = await fetch(`${API_BASE_URL}/digitize`, {
      method: 'POST',
      body: fd
    });
    if (!res.ok) throw new Error(`Digitize failed: ${res.statusText}`);
    return res.json();
  },

  async getDigitizeStatus() {
    const res = await fetch(`${API_BASE_URL}/digitize-status`);
    if (!res.ok) throw new Error(`Digitize status failed: ${res.statusText}`);
    return res.json();
  },

  async getDigitizeResults(sessionId) {
    const res = await fetch(`${API_BASE_URL}/digitize-results/${sessionId}`);
    if (!res.ok) throw new Error(`Results fetch failed: ${res.statusText}`);
    return res.json();
  },

  async downloadDigitized(sessionId, format = 'txt') {
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('format', format);
    const res = await fetch(`${API_BASE_URL}/digitize/download`, {
      method: 'POST',
      body: fd
    });
    if (!res.ok) throw new Error(`Download failed: ${res.statusText}`);
    return res.blob();
  },

  getDigitizePageImageUrl(sessionId, pageNum) {
    return `${API_BASE_URL}/digitize-page-image/${sessionId}/${pageNum}`;
  },

  getDigitizeOutputFileUrl(filename) {
    return `${API_BASE_URL}/digitize-output-file/${encodeURIComponent(filename)}`;
  }
};

export default api;

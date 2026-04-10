const API_BASE_URL = 'http://127.0.0.1:8000';

const api = {
  async ingestFiles(files) {
    const fd = new FormData();
    for (const f of files) {
      fd.append('files', f);
    }
    fd.append('reset_db', 'false'); // Don't reset the database when adding new files
    const res = await fetch(`${API_BASE_URL}/ingest`, { 
      method: 'POST', 
      body: fd 
    });
    if (!res.ok) throw new Error(`Ingest failed: ${res.statusText}`);
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
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;
          const data = trimmed.slice(6); // Remove 'data: '

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
    return res.blob(); // Return blob for audio file
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
    // Routes through the preview endpoint which converts DOCX/TXT/CSV to HTML
    // and redirects images/PDFs to direct serving
    return `${API_BASE_URL}/files/${encodeURIComponent(filename)}/preview`;
  },

  async listFiles() {
    const res = await fetch(`${API_BASE_URL}/files`);
    if (!res.ok) throw new Error(`List files failed: ${res.statusText}`);
    return res.json();
  }
};

export default api;

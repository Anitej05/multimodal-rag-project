const API_BASE_URL = 'http://127.0.0.1:8000';

const api = {
  async ingestFiles(files) {
    const fd = new FormData();
    for (const f of files) {
      fd.append('files', f);
    }
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

  async resetKB() {
    const res = await fetch(`${API_BASE_URL}/reset`, { 
      method: 'POST' 
    });
    if (!res.ok) throw new Error(`Reset failed: ${res.statusText}`);
    return res.json();
  }
};

export default api;
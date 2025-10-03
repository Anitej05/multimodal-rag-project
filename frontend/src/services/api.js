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
  }
};

export default api;

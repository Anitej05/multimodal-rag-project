# Multimodal RAG Knowledge Base Assistant

A modern React application for managing and querying a multimodal knowledge base with RAG (Retrieval Augmented Generation) capabilities.

## Features

- 📚 **Knowledge Base Management**: Upload and index multiple file types
- 💬 **AI Chat Assistant**: Ask questions about your uploaded documents
- 🎤 **Audio Support**: Upload audio files or record directly in-browser
- 🖼️ **Image Support**: Visual file previews for images
- 📊 **Statistics Dashboard**: Track files, indexing status, and storage
- 🎨 **Modern UI**: Sleek, professional dark-themed interface

## Supported File Types

- **Documents**: PDF, TXT, DOCX, CSV
- **Images**: PNG, JPG, JPEG
- **Audio**: MP3, WAV, M4A, WebM

## Directory Structure

```
multimodal-rag-react/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Header.jsx
│   │   ├── KnowledgeBase.jsx
│   │   ├── Chat.jsx
│   │   ├── FileUpload.jsx
│   │   ├── FileList.jsx
│   │   ├── Toast.jsx
│   │   └── StatsCard.jsx
│   ├── services/
│   │   └── api.js
│   ├── utils/
│   │   └── helpers.js
│   ├── styles/
│   │   └── globals.css
│   ├── App.jsx
│   └── index.js
├── package.json
└── README.md
```

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend API running on `http://127.0.0.1:8000` with the following endpoints:
  - `POST /ingest` - Upload and index files
  - `POST /chat` - Text-based chat
  - `POST /chat-audio` - Audio-based chat
  - `POST /reset` - Reset knowledge base

## Installation

1. **Clone or create the project directory:**
   ```bash
   mkdir multimodal-rag-react
   cd multimodal-rag-react
   ```

2. **Copy all the files into their respective directories as shown in the structure above**

3. **Install dependencies:**
   ```bash
   npm install
   ```

## Running the Application

1. **Make sure your backend API is running on `http://127.0.0.1:8000`**

2. **Start the development server:**
   ```bash
   npm start
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:3000
   ```

## Usage

### Uploading Files

1. Navigate to the **Upload** tab in the Knowledge Base section
2. Either drag and drop files or click to browse
3. View file statistics in real-time
4. Click **Index Knowledge Base** to process the files

### Chatting with Your Knowledge Base

1. Once files are indexed, navigate to the AI Assistant section
2. Type your question in the input field
3. Press Enter or click the send button
4. View AI responses in real-time

### Audio Features

- **Upload Audio**: Click the microphone icon to attach an audio file
- **Record Audio**: Click the record button to record directly from your microphone
- Send the audio message and receive AI responses

### Managing Files

- Switch to the **Files** tab to view all uploaded files
- See file status (pending/indexed)
- Use the **Reset** button to clear the entire knowledge base

## API Configuration

To change the backend API URL, edit `src/services/api.js`:

```javascript
const API_BASE_URL = 'http://your-api-url:port';
```

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` directory.

## Technologies Used

- **React 18** - UI framework
- **CSS3** - Styling with custom properties and animations
- **Fetch API** - HTTP requests
- **MediaRecorder API** - Audio recording
- **FileReader API** - File handling

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

**Note**: Audio recording requires HTTPS in production or localhost in development.

## Troubleshooting

### CORS Errors
If you encounter CORS errors, ensure your backend API has proper CORS headers configured.

### Audio Recording Not Working
- Check browser permissions for microphone access
- Ensure you're using HTTPS (or localhost for development)

### Files Not Uploading
- Verify the backend API is running
- Check file size limits on your backend
- Ensure file types are supported

## License

MIT License

## Support

For issues and questions, please contact your development team or create an issue in the project repository.
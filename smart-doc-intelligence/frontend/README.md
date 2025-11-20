# Smart Document Intelligence Platform - Web UI

Beautiful Streamlit web interface for the Smart Document Intelligence Platform.

## 🌟 Features

- **📤 Document Upload**: Upload and process PDFs and images
- **💬 Interactive Chat**: Ask questions with RAG-powered answers
- **📚 Document Browser**: View and manage your document library
- **📊 Advanced Analysis**:
  - Entity extraction (emails, phones, dates, organizations, etc.)
  - Document summarization with multiple styles and lengths
  - Document comparison and similarity analysis
  - Academic citation generation (APA, MLA, Chicago)
  - Export to JSON, Markdown, and HTML
- **⚙️ Settings**: Configure LLMs, vector database, and OCR options

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- All backend dependencies installed
- Ollama running (optional, for local LLM)
- Gemini API key (optional, for cloud LLM)

### Installation

```bash
# Navigate to the smart-doc-intelligence directory
cd smart-doc-intelligence

# Streamlit is already in requirements.txt
# If not installed, run:
pip install streamlit streamlit-extras

# Run the app
streamlit run frontend/app.py
```

### First Run

1. The app will open in your browser at `http://localhost:8501`
2. Click **📤 Upload Documents** to add your first document
3. Upload a PDF or image file
4. Wait for processing to complete
5. Navigate to **💬 Chat & Q&A** to ask questions!

## 📖 User Guide

### Uploading Documents

1. Go to **📤 Upload Documents**
2. Click "Browse files" or drag and drop
3. Select processing options:
   - **Prompt Type**: document, free, figure, or detail
   - **Enhance Images**: Apply image preprocessing
   - **Index in Vector Database**: Enable semantic search
   - **Chunking Strategy**: How to split text
4. Click **🚀 Process Documents**
5. Wait for processing (1-3 seconds per page)

### Asking Questions

1. Go to **💬 Chat & Q&A**
2. Type your question in the chat input
3. Options:
   - **Collection**: Select which documents to search
   - **LLM**: Auto, Local (Ollama), or Cloud (Gemini)
   - **Number of sources**: How many chunks to retrieve
   - **Temperature**: Creativity level
   - **Show sources**: Display source documents
4. View the answer with source citations

### Browsing Documents

1. Go to **📚 Document Library**
2. View options:
   - **Cards**: Visual card layout
   - **List**: Detailed list view
   - **Table**: Tabular format
3. Filters:
   - Status (uploaded, processed, error)
   - Type (PDF, image)
   - Search by filename
   - Sort options
4. Actions per document:
   - **View**: See full details and text
   - **Chat**: Ask questions about specific document
   - **Analyze**: Run advanced analysis

### Document Analysis

Go to **📊 Analysis** and select from:

#### 🔍 Entity Extraction
- Extract emails, phones, URLs, dates, money, percentages
- Use LLM for people, organizations, locations
- Extract key terms with frequency

#### 📝 Summarization
- **Styles**: paragraph, bullet points, executive, technical, simple
- **Lengths**: very short, short, medium, long, detailed
- **Methods**: auto, extractive, abstractive, hierarchical
- Use local or cloud LLM

#### 🔄 Document Comparison
- Compare 2-5 documents
- Lexical and semantic similarity
- Common and unique terms
- LLM-generated comparison summary

#### 📚 Citation Generator
- Styles: APA (7th), MLA (9th), Chicago (17th)
- Document types: journal, book, article, website, conference, thesis
- Fill in metadata and generate formatted citations

#### 📤 Export
- Export to JSON, Markdown, or HTML
- Include entities and summaries
- Download formatted results

## ⚙️ Configuration

### Settings Page

Access via **⚙️ Settings** to configure:

**LLM Settings**:
- Ollama server URL and model
- Gemini API key and model
- Test connections

**Vector Database**:
- ChromaDB storage path
- Embedding model selection

**OCR Settings**:
- Base resolution (512, 640, 1024, 1280)
- Crop mode enable/disable
- Maximum crops

### Environment Variables

Create `.env` in the `smart-doc-intelligence` directory:

```env
# Required for OCR
DEEPSEEK_MODEL_PATH=deepseek-ai/DeepSeek-OCR
CUDA_VISIBLE_DEVICES=0

# Optional for Gemini
GEMINI_API_KEY=your-api-key-here

# Optional for Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3
```

## 🎨 UI Components

### Main Pages

- **🏠 Home**: Overview, features, quick start, statistics
- **📤 Upload**: File upload with processing options
- **💬 Chat**: Interactive Q&A with RAG
- **📚 Documents**: Library browser with filters and views
- **📊 Analysis**: All Phase 4 features in tabs
- **⚙️ Settings**: System configuration

### Navigation

- Sidebar navigation buttons
- Quick actions for common tasks
- Breadcrumb-style page flow

### UI Features

- Beautiful gradient headers
- Responsive card layouts
- Interactive expanders
- Progress bars for processing
- Download buttons for exports
- Metrics and statistics
- Color-coded status badges

## 🔧 Customization

### Styling

Edit the CSS in `app.py` to customize colors, fonts, and layout.

### Adding Pages

1. Create new page in `frontend/pages/`
2. Import and call in `app.py`
3. Add navigation button in sidebar

### Custom Components

Create reusable components in `frontend/components/` (future expansion).

## 📊 Performance Tips

### For Faster Processing

1. Reduce OCR resolution for simple documents
2. Disable vector indexing if not needed
3. Use smaller chunking sizes
4. Use local LLM (Ollama) for speed

### For Better Quality

1. Use higher OCR resolution (1280px)
2. Enable image enhancement
3. Use cloud LLM (Gemini) for complex queries
4. Increase number of sources in RAG

## 🐛 Troubleshooting

### App Won't Start

```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Run with verbose output
streamlit run frontend/app.py --logger.level=debug
```

### Processing Fails

- Check backend dependencies are installed
- Verify GPU is available for OCR
- Check file permissions on storage directory

### Chat Not Working

- Ensure documents are processed and indexed
- Check LLM is running (Ollama or Gemini API key)
- Verify ChromaDB is accessible

### UI Looks Broken

```bash
# Clear Streamlit cache
streamlit cache clear

# Update Streamlit
pip install --upgrade streamlit
```

## 📱 Mobile Support

The UI is responsive and works on tablets. For best experience, use desktop with a larger screen.

## 🔐 Security Notes

- Local mode (Ollama) keeps all data on your machine
- Cloud mode (Gemini) sends queries to Google AI
- Documents are stored locally in `storage/`
- No data is sent to third parties except when using Gemini

## 📝 Keyboard Shortcuts

- `Ctrl+K`: Focus search
- `Enter`: Submit chat message
- `R`: Rerun app

## 🎯 Tips & Tricks

1. **Use suggested questions** in chat for quick start
2. **Filter documents** by status before analysis
3. **Export results** before leaving analysis page
4. **Save citations** for later use
5. **Test LLM connections** in settings before first use

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Phase 5 Complete](../../PHASE5_COMPLETE.md)
- [Project README](../README.md)

## 🤝 Contributing

UI improvements welcome:
- Better visualizations
- Additional chart types
- Mobile optimizations
- New page layouts
- Component library

---

**Built with Streamlit** | **Powered by Smart Document Intelligence Platform**

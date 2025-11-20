# Smart Document Intelligence Platform

A powerful document processing system that combines DeepSeek-OCR, local LLMs (Ollama), and cloud AI (Gemini) for intelligent document understanding.

## 🌟 Features

### Phase 1: Core OCR Pipeline (✅ Complete)
- **Multi-format Support**: PDF and image document processing
- **Advanced OCR**: DeepSeek-OCR with layout preservation
- **Smart Chunking**: Multiple chunking strategies (paragraph, sentence, markdown)
- **Document Storage**: Organized file management with metadata
- **Batch Processing**: Process multiple documents efficiently

### Phase 2: Vector Database & RAG (✅ Complete)
- **ChromaDB Integration**: Vector storage with collections and metadata
- **Semantic Embeddings**: sentence-transformers for text embeddings
- **Document Retrieval**: Similarity search with scoring and filtering
- **RAG Engine**: Context retrieval and prompt building

### Phase 3: LLM Integration (✅ Complete)
- **Dual LLM System**: Ollama (local) + Gemini (cloud)
- **Query Routing**: Intelligent LLM selection (privacy vs performance)
- **Answer Generation**: Context-aware responses
- **Conversational RAG**: Multi-turn dialogue support

### Phase 4: Advanced Features (✅ Complete)
- **Entity Extraction**: Pattern-based and LLM-based NER
- **Document Comparison**: Similarity analysis and diff generation
- **Auto-Summarization**: Extractive, abstractive, and hierarchical strategies
- **Export Functionality**: JSON, Markdown, and HTML export
- **Citation Generator**: APA, MLA, and Chicago style citations

### Phase 5: Web Interface (Coming Soon)
- Streamlit web UI
- Document upload interface
- Interactive chat with RAG
- Visualization and analytics
- Export options in UI

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for DeepSeek-OCR)
- GPU with 24GB+ VRAM (recommended: RTX 4090, A100)
- 16GB+ RAM

### Installation

```bash
# 1. Clone repository
cd DeepSeek-OCR/smart-doc-intelligence

# 2. Create virtual environment
conda create -n smart-doc python=3.12 -y
conda activate smart-doc

# 3. Install PyTorch with CUDA
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install vLLM (download wheel from releases page)
# https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 5. Install other dependencies
pip install -r requirements.txt

# 6. Install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation

# 7. Set up environment variables
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Create a `.env` file:

```env
DEEPSEEK_MODEL_PATH=deepseek-ai/DeepSeek-OCR
CUDA_VISIBLE_DEVICES=0
GEMINI_API_KEY=your-api-key-here
```

### Basic Usage

#### 1. Test the Pipeline

```bash
# Run tests
python tests/test_phase1_pipeline.py
```

#### 2. Process Documents via Python API

```python
from backend.pipeline import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline(load_ocr_model=True)

# Process a PDF
result = pipeline.process_pdf("document.pdf")

# Process an image
result = pipeline.process_image("scan.jpg", enhance=True)

# List all documents
docs = pipeline.list_documents()

# Get statistics
stats = pipeline.get_statistics()
```

#### 3. Process Documents via CLI

```bash
# Load model and process a PDF
python backend/pipeline.py --load-model --file document.pdf --type document

# List all processed documents
python backend/pipeline.py --list

# Show statistics
python backend/pipeline.py --stats
```

## 📁 Project Structure

```
smart-doc-intelligence/
├── backend/
│   ├── ocr/
│   │   ├── deepseek_wrapper.py      # DeepSeek-OCR interface
│   │   ├── pdf_processor.py         # PDF to image conversion
│   │   └── image_processor.py       # Image preprocessing
│   ├── vectordb/                    # [Phase 2] Vector database
│   ├── llm/                         # [Phase 3] LLM integration
│   ├── features/                    # [Phase 4] Advanced features
│   ├── utils/
│   │   ├── config.py                # Configuration management
│   │   ├── chunking.py              # Text chunking
│   │   └── storage.py               # File storage system
│   └── pipeline.py                  # Main processing pipeline
├── frontend/                        # [Phase 5] Streamlit UI
├── storage/
│   ├── uploads/                     # Original files
│   ├── processed/                   # Extracted text & chunks
│   └── metadata/                    # Document metadata
├── tests/
│   └── test_phase1_pipeline.py      # Phase 1 tests
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration Options

### DeepSeek-OCR Settings

```python
# config.py or environment variables
DEEPSEEK_MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
BASE_SIZE = 1024          # Resolution: 512, 640, 1024, 1280
IMAGE_SIZE = 640          # Crop size for dynamic mode
CROP_MODE = True          # Enable Gundam mode (dynamic cropping)
MAX_CROPS = 6             # Max crops (reduce if low GPU memory)
```

### OCR Prompts

Different prompts for different use cases:

- **Document → Markdown**: `<image>\n<|grounding|>Convert the document to markdown.`
- **Free OCR**: `<image>\nFree OCR.`
- **Figure Parsing**: `<image>\nParse the figure.`
- **Detailed Description**: `<image>\nDescribe this image in detail.`

### Chunking Strategies

```python
chunker = DocumentChunker(
    chunk_size=500,        # Target chunk size (characters)
    chunk_overlap=100,     # Overlap between chunks
    strategy="paragraph"   # 'fixed', 'paragraph', 'sentence', 'markdown'
)
```

## 📊 Components Overview

### 1. DeepSeek-OCR Wrapper

Clean interface to DeepSeek-OCR with:
- Sync and async inference
- Batch processing
- Multiple prompt types
- Automatic image preprocessing

```python
from backend.ocr.deepseek_wrapper import DeepSeekOCR

ocr = DeepSeekOCR()
ocr.load_model(batch_mode=False)

result = ocr.extract_text("image.jpg", prompt_type="document")
print(result["text"])
```

### 2. PDF Processor

Convert PDFs to images with control over quality:

```python
from backend.ocr.pdf_processor import PDFProcessor

processor = PDFProcessor(dpi=144)  # 144 DPI = good quality
images = processor.pdf_to_images("document.pdf")

# Save images
processor.save_images(images, "output_dir", format="PNG")
```

### 3. Image Processor

Enhance images for better OCR:

```python
from backend.ocr.image_processor import ImageProcessor

processor = ImageProcessor()
img = processor.load_image("scan.jpg")

# Auto-enhance for OCR
enhanced = processor.enhance_for_ocr(
    img,
    auto_contrast=True,
    sharpen=True,
    denoise=False
)
```

### 4. Document Chunker

Split text into semantic chunks:

```python
from backend.utils.chunking import DocumentChunker

chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)

# Paragraph-aware chunking
chunks = chunker.chunk_by_paragraph(text)

# Markdown section chunking
chunks = chunker.chunk_by_markdown_section(markdown_text)
```

### 5. Storage System

Manage documents and metadata:

```python
from backend.utils.storage import DocumentStorage

storage = DocumentStorage()

# Save upload
result = storage.save_upload("document.pdf", doc_type="pdf")
doc_id = result["doc_id"]

# Save processed results
storage.save_processed(doc_id, extracted_text, chunks)

# Retrieve
text = storage.get_processed_text(doc_id)
chunks = storage.get_chunks(doc_id)

# List all documents
docs = storage.list_documents(status="processed")
```

## 🎯 Use Cases

1. **Document Digitization**: Convert scanned documents to searchable text
2. **Invoice Processing**: Extract data from invoices and receipts
3. **Contract Analysis**: Parse legal documents for review
4. **Research Paper Processing**: Extract and organize academic papers
5. **Form Processing**: Digitize forms and extract structured data
6. **Book Scanning**: Convert physical books to digital format

## ⚙️ Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| PDF Upload (10 pages) | 2-5s | DeepSeek-OCR processing |
| Single page OCR | 1-3s | 1024x1024 resolution |
| Batch OCR (PDF) | ~2500 tokens/s | A100-40G GPU |
| Image enhancement | <0.5s | CPU processing |
| Chunking | <0.1s | Per 10k characters |

## 🔮 Roadmap

### Phase 1: Core OCR Pipeline ✅
- [x] DeepSeek-OCR integration
- [x] PDF/image processing
- [x] Smart chunking strategies
- [x] Document storage system

### Phase 2: Vector Database & RAG ✅
- [x] ChromaDB integration
- [x] Embedding generation
- [x] Semantic search
- [x] Document retrieval
- [x] RAG engine

### Phase 3: Dual LLM Integration ✅
- [x] Ollama setup (Llama 3.3, Mistral)
- [x] Gemini API integration
- [x] Query routing logic
- [x] Response synthesis
- [x] Conversational RAG

### Phase 4: Advanced Features ✅
- [x] Document comparison engine
- [x] Entity extraction (pattern + LLM)
- [x] Auto-summarization (extractive, abstractive, hierarchical)
- [x] Export functionality (JSON, Markdown, HTML)
- [x] Citation generator (APA, MLA, Chicago)

### Phase 5: Web Interface (Next) 🚧
- [ ] Streamlit UI framework
- [ ] Document upload interface
- [ ] Interactive chat with RAG
- [ ] Query interface with streaming
- [ ] Document viewer with highlighting
- [ ] Results visualization
- [ ] Export options in UI

## 🐛 Troubleshooting

### GPU Memory Issues

```python
# Reduce max_crops in config
MAX_CROPS = 4  # Instead of 6

# Reduce GPU utilization
gpu_memory_utilization = 0.6  # Instead of 0.75
```

### vLLM Installation Issues

If vLLM fails to install:
1. Download the correct wheel for your Python version
2. Ensure CUDA 11.8 is installed
3. Check GPU compatibility

### Image Quality Issues

```python
# Increase DPI for better quality
processor = PDFProcessor(dpi=300)  # Instead of 144

# Enable image enhancement
enhanced = processor.enhance_for_ocr(img, auto_contrast=True, sharpen=True)
```

## 📝 License

This project uses DeepSeek-OCR and other open-source components. Check individual licenses:
- DeepSeek-OCR: [License](https://github.com/deepseek-ai/DeepSeek-OCR)
- vLLM: Apache 2.0

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional chunking strategies
- OCR prompt optimization
- UI/UX enhancements
- Performance optimizations

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review test scripts for examples
3. Open an issue on GitHub

---

**Status**: Phases 1-4 Complete ✅ | Phase 5 (UI) Next 🚧

**Built with**: DeepSeek-OCR, vLLM, ChromaDB, Ollama, Gemini, PyMuPDF, sentence-transformers, and more.

## 📚 Additional Documentation

- [Phase 1 Complete](../PHASE1_COMPLETE.md) - Core OCR Pipeline
- [Phase 2 Complete](../PHASE2_COMPLETE.md) - Vector Database & RAG
- [Phase 3 Complete](../PHASE3_COMPLETE.md) - LLM Integration
- [Phase 4 Complete](../PHASE4_COMPLETE.md) - Advanced Features

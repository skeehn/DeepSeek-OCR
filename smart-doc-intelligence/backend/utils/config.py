"""
Configuration management for Smart Document Intelligence Platform
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
CHROMA_DB_DIR = STORAGE_DIR / "chroma_db"
MODELS_DIR = BASE_DIR / "models"


class DeepSeekOCRConfig(BaseModel):
    """DeepSeek-OCR configuration"""
    model_path: str = Field(
        default=os.getenv("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR"),
        description="Path to DeepSeek-OCR model"
    )

    # Resolution modes: tiny=512, small=640, base=1024, large=1280
    base_size: int = Field(default=1024, description="Base resolution for OCR")
    image_size: int = Field(default=640, description="Image crop size")
    crop_mode: bool = Field(default=True, description="Enable dynamic cropping (Gundam mode)")

    min_crops: int = Field(default=2, description="Minimum number of crops")
    max_crops: int = Field(default=6, description="Maximum crops (reduce if low GPU memory)")

    max_tokens: int = Field(default=8192, description="Max output tokens")
    temperature: float = Field(default=0.0, description="Sampling temperature")

    # vLLM settings
    gpu_memory_utilization: float = Field(default=0.75, description="GPU memory utilization")
    max_model_len: int = Field(default=8192, description="Max model context length")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism")

    # Prompts
    document_prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    free_ocr_prompt: str = "<image>\nFree OCR."
    figure_prompt: str = "<image>\nParse the figure."
    detail_prompt: str = "<image>\nDescribe this image in detail."

    cuda_device: str = Field(default="0", description="CUDA device ID")


class GeminiConfig(BaseModel):
    """Gemini API configuration"""
    api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API key"
    )
    model_name: str = Field(default="gemini-2.0-flash", description="Gemini model version")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_output_tokens: int = Field(default=8192, description="Max output tokens")

    # Rate limiting
    requests_per_minute: int = Field(default=15, description="RPM limit for free tier")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=2.0, description="Delay between retries (seconds)")


class OllamaConfig(BaseModel):
    """Ollama local LLM configuration"""
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model_name: str = Field(default="llama3.3", description="Ollama model to use")
    temperature: float = Field(default=0.3, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Max output tokens")

    # Alternative models
    fallback_model: str = Field(default="mistral", description="Fallback model if primary fails")


class ChromaDBConfig(BaseModel):
    """ChromaDB vector database configuration"""
    persist_directory: str = Field(
        default=str(CHROMA_DB_DIR),
        description="ChromaDB persistence directory"
    )
    collection_name: str = Field(default="documents", description="Default collection name")

    # Embedding model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    # Search parameters
    top_k: int = Field(default=5, description="Number of results to retrieve")
    score_threshold: float = Field(default=0.5, description="Minimum similarity score")


class ChunkingConfig(BaseModel):
    """Text chunking configuration"""
    chunk_size: int = Field(default=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")

    # Chunking strategy: 'fixed', 'paragraph', 'semantic'
    strategy: str = Field(default="paragraph", description="Chunking strategy")

    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int = Field(default=1000, description="Maximum chunk size")


class StorageConfig(BaseModel):
    """File storage configuration"""
    uploads_dir: Path = Field(default=UPLOADS_DIR, description="Uploads directory")
    processed_dir: Path = Field(default=PROCESSED_DIR, description="Processed files directory")

    # File size limits
    max_file_size_mb: int = Field(default=50, description="Max upload file size in MB")

    # Supported formats
    supported_image_formats: list = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        description="Supported image formats"
    )
    supported_document_formats: list = Field(
        default=[".pdf"],
        description="Supported document formats"
    )

    # Storage options
    compress_processed: bool = Field(default=False, description="Compress processed files")
    retention_days: int = Field(default=90, description="File retention period")


class AppConfig(BaseModel):
    """Main application configuration"""
    app_name: str = "Smart Document Intelligence Platform"
    version: str = "0.1.0"
    debug: bool = Field(default=False, description="Debug mode")

    # Component configs
    deepseek: DeepSeekOCRConfig = Field(default_factory=DeepSeekOCRConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get global configuration instance"""
    return config


def update_config(**kwargs):
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        STORAGE_DIR,
        UPLOADS_DIR,
        PROCESSED_DIR,
        CHROMA_DB_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"✅ Created directories in: {BASE_DIR}")


if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print(f"Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"DeepSeek model path: {config.deepseek.model_path}")
    print(f"Gemini API key set: {'Yes' if config.gemini.api_key else 'No'}")

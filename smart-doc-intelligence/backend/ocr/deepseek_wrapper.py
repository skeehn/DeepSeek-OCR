"""
DeepSeek-OCR Wrapper for Smart Document Intelligence Platform
Provides a clean interface to DeepSeek-OCR functionality
"""
import os
import sys
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image, ImageOps
import torch

# Add DeepSeek-OCR to path
DEEPSEEK_OCR_PATH = Path(__file__).parent.parent.parent.parent / "DeepSeek-OCR-master/DeepSeek-OCR-vllm"
sys.path.insert(0, str(DEEPSEEK_OCR_PATH))

try:
    from vllm import LLM, SamplingParams, AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
except ImportError as e:
    print(f"Warning: DeepSeek-OCR dependencies not found: {e}")
    print("Make sure vLLM and DeepSeek-OCR are properly installed.")

from backend.utils.config import get_config


class DeepSeekOCR:
    """
    Wrapper class for DeepSeek-OCR model
    Supports both sync and async inference
    """

    def __init__(self, config=None):
        """
        Initialize DeepSeek-OCR wrapper

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config().deepseek
        self.llm = None
        self.async_engine = None
        self.processor = DeepseekOCRProcessor()

        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_device

        if torch.version.cuda == '11.8':
            os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

        os.environ['VLLM_USE_V1'] = '0'

        # Register model
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

        print(f"✅ DeepSeek-OCR wrapper initialized")
        print(f"   Model: {self.config.model_path}")
        print(f"   Resolution: {self.config.base_size}x{self.config.base_size}")
        print(f"   Crop mode: {self.config.crop_mode}")

    def load_model(self, batch_mode: bool = False):
        """
        Load the DeepSeek-OCR model

        Args:
            batch_mode: If True, load for batch processing (higher concurrency)
        """
        print("🔄 Loading DeepSeek-OCR model...")

        try:
            self.llm = LLM(
                model=self.config.model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=True,
                max_model_len=self.config.max_model_len,
                swap_space=0,
                max_num_seqs=100 if batch_mode else 1,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                disable_mm_preprocessor_cache=True if batch_mode else False,
            )
            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    async def load_model_async(self):
        """Load model for async inference"""
        print("🔄 Loading DeepSeek-OCR model (async)...")

        try:
            engine_args = AsyncEngineArgs(
                model=self.config.model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                max_model_len=self.config.max_model_len,
                enforce_eager=False,
                trust_remote_code=True,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("✅ Async model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading async model: {e}")
            return False

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess an image

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object or None if error
        """
        try:
            image = Image.open(image_path)
            # Correct orientation based on EXIF data
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image.convert('RGB')

        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    def preprocess_image(self, image: Image.Image) -> Any:
        """
        Preprocess image for DeepSeek-OCR

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image features
        """
        image_features = self.processor.tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=self.config.crop_mode
        )
        return image_features

    def extract_text(
        self,
        image_path: str,
        prompt_type: str = "document",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text from a single image (synchronous)

        Args:
            image_path: Path to image file
            prompt_type: Type of prompt ('document', 'free', 'figure', 'detail')
            custom_prompt: Custom prompt (overrides prompt_type)

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.llm:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}

        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompts_map = {
                "document": self.config.document_prompt,
                "free": self.config.free_ocr_prompt,
                "figure": self.config.figure_prompt,
                "detail": self.config.detail_prompt,
            }
            prompt = prompts_map.get(prompt_type, self.config.document_prompt)

        # Preprocess image
        image_features = self.preprocess_image(image)

        # Prepare sampling parameters
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}  # <td>, </td>
            )
        ]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        # Prepare request
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }

        # Generate
        try:
            outputs = self.llm.generate([request], sampling_params=sampling_params)
            result_text = outputs[0].outputs[0].text

            return {
                "success": True,
                "text": result_text,
                "prompt": prompt,
                "image_path": image_path,
                "image_size": image.size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }

    async def extract_text_async(
        self,
        image_path: str,
        prompt_type: str = "document",
        custom_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from a single image (asynchronous, with optional streaming)

        Args:
            image_path: Path to image file
            prompt_type: Type of prompt
            custom_prompt: Custom prompt
            stream: Enable streaming output

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.async_engine:
            raise RuntimeError("Async engine not loaded. Call load_model_async() first.")

        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}

        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompts_map = {
                "document": self.config.document_prompt,
                "free": self.config.free_ocr_prompt,
                "figure": self.config.figure_prompt,
                "detail": self.config.detail_prompt,
            }
            prompt = prompts_map.get(prompt_type, self.config.document_prompt)

        # Preprocess
        image_features = self.preprocess_image(image)

        # Sampling params
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        # Request
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }

        request_id = f"request-{hash(image_path)}"

        try:
            full_text = ""
            async for request_output in self.async_engine.generate(
                request, sampling_params, request_id
            ):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text

                    if stream:
                        # For streaming, you'd yield here
                        print(full_text[-100:], end='', flush=True)

            return {
                "success": True,
                "text": full_text,
                "prompt": prompt,
                "image_path": image_path,
                "image_size": image.size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }

    def batch_extract(
        self,
        image_paths: List[str],
        prompt_type: str = "document"
    ) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images in batch

        Args:
            image_paths: List of image paths
            prompt_type: Type of prompt to use

        Returns:
            List of result dictionaries
        """
        if not self.llm:
            raise RuntimeError("Model not loaded. Call load_model(batch_mode=True) first.")

        # Get prompt
        prompts_map = {
            "document": self.config.document_prompt,
            "free": self.config.free_ocr_prompt,
            "figure": self.config.figure_prompt,
            "detail": self.config.detail_prompt,
        }
        prompt = prompts_map.get(prompt_type, self.config.document_prompt)

        # Prepare batch inputs
        batch_inputs = []
        valid_images = []

        for img_path in image_paths:
            image = self.load_image(img_path)
            if image:
                image_features = self.preprocess_image(image)
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_features}
                })
                valid_images.append((img_path, image.size))

        if not batch_inputs:
            return [{"success": False, "error": "No valid images to process"}]

        # Sampling params
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=20,
                window_size=50,
                whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        # Generate
        try:
            outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)

            results = []
            for output, (img_path, img_size) in zip(outputs, valid_images):
                result_text = output.outputs[0].text

                results.append({
                    "success": True,
                    "text": result_text,
                    "prompt": prompt,
                    "image_path": img_path,
                    "image_size": img_size,
                })

            return results

        except Exception as e:
            return [{
                "success": False,
                "error": str(e),
                "batch_size": len(image_paths)
            }]


# Convenience functions
def create_ocr_engine(batch_mode: bool = False) -> DeepSeekOCR:
    """
    Create and load a DeepSeek-OCR engine

    Args:
        batch_mode: Whether to enable batch processing

    Returns:
        Loaded DeepSeekOCR instance
    """
    ocr = DeepSeekOCR()
    ocr.load_model(batch_mode=batch_mode)
    return ocr


async def create_ocr_engine_async() -> DeepSeekOCR:
    """Create and load async DeepSeek-OCR engine"""
    ocr = DeepSeekOCR()
    await ocr.load_model_async()
    return ocr

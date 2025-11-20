"""
Gemini API Client Wrapper
Handles Google Gemini API interactions
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")

from backend.utils.config import get_config


@dataclass
class GeminiResponse:
    """Represents a Gemini API response"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
        }


class GeminiClient:
    """
    Client for interacting with Google Gemini API
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini client

        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-2.0-flash)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        config = get_config().gemini

        self.api_key = api_key or config.api_key
        self.model_name = model or config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_output_tokens
        self.retry_attempts = config.retry_attempts
        self.retry_delay = config.retry_delay

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")

        # Configure API
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)

        print(f"✅ Gemini client initialized")
        print(f"   Model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> GeminiResponse:
        """
        Generate text using Gemini

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: System instruction

        Returns:
            GeminiResponse object
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Build generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Add system instruction if provided
        if system:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system
            )
        else:
            model = self.model

        # Retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Extract text
                text = response.text if hasattr(response, 'text') else ""

                # Extract token counts
                prompt_tokens = 0
                completion_tokens = 0

                if hasattr(response, 'usage_metadata'):
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    completion_tokens = response.usage_metadata.candidates_token_count

                return GeminiResponse(
                    text=text,
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    finish_reason=response.candidates[0].finish_reason.name if response.candidates else "stop"
                )

            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Gemini generation error: {e}")
                    return GeminiResponse(
                        text=f"Error: {str(e)}",
                        model=self.model_name
                    )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeminiResponse:
        """
        Chat with Gemini using message history

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            GeminiResponse object
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Start chat session
        chat = self.model.start_chat(history=[])

        # Add message history (all except last)
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            chat.history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Send last message
        last_message = messages[-1]["content"]

        try:
            response = chat.send_message(
                last_message,
                generation_config=generation_config
            )

            text = response.text if hasattr(response, 'text') else ""

            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

            return GeminiResponse(
                text=text,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else "stop"
            )

        except Exception as e:
            print(f"❌ Gemini chat error: {e}")
            return GeminiResponse(
                text=f"Error: {str(e)}",
                model=self.model_name
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            print(f"❌ Error counting tokens: {e}")
            return 0

    def list_models(self) -> List[str]:
        """
        List available Gemini models

        Returns:
            List of model names
        """
        try:
            models = genai.list_models()
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []


class GeminiRateLimiter:
    """
    Simple rate limiter for Gemini API
    Handles free tier limits (15 RPM)
    """

    def __init__(self, requests_per_minute: int = 15):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

        print(f"⏱️ Rate limiter initialized: {requests_per_minute} RPM")

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            print(f"⏳ Rate limiting: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

        self.last_request_time = time.time()


class RateLimitedGeminiClient(GeminiClient):
    """Gemini client with automatic rate limiting"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize rate-limited Gemini client"""
        super().__init__(api_key=api_key, model=model)

        config = get_config().gemini
        self.rate_limiter = GeminiRateLimiter(
            requests_per_minute=config.requests_per_minute
        )

    def generate(self, prompt: str, **kwargs) -> GeminiResponse:
        """Generate with rate limiting"""
        self.rate_limiter.wait_if_needed()
        return super().generate(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> GeminiResponse:
        """Chat with rate limiting"""
        self.rate_limiter.wait_if_needed()
        return super().chat(messages, **kwargs)


# Convenience functions
def create_gemini_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    rate_limited: bool = True
) -> GeminiClient:
    """
    Create Gemini client

    Args:
        api_key: API key
        model: Model name
        rate_limited: Use rate-limited client

    Returns:
        GeminiClient instance
    """
    if rate_limited:
        return RateLimitedGeminiClient(api_key=api_key, model=model)
    else:
        return GeminiClient(api_key=api_key, model=model)


def quick_generate(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Quick text generation

    Args:
        prompt: Input prompt
        api_key: API key

    Returns:
        Generated text
    """
    client = GeminiClient(api_key=api_key)
    response = client.generate(prompt)
    return response.text


if __name__ == "__main__":
    # Test Gemini client
    print("Gemini Client Test")
    print("=" * 50)

    try:
        client = create_gemini_client(rate_limited=True)

        # Test generation
        print("\nTesting generation:")
        response = client.generate(
            "Explain what is artificial intelligence in one sentence.",
            temperature=0.7
        )

        print(f"Response: {response.text}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Finish reason: {response.finish_reason}")

        # Test token counting
        token_count = client.count_tokens("This is a test sentence.")
        print(f"\nToken count test: {token_count} tokens")

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("   Set GEMINI_API_KEY environment variable")
    except Exception as e:
        print(f"❌ Error: {e}")

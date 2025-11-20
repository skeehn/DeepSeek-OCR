"""
Ollama Client Wrapper
Handles local LLM inference using Ollama
"""
import json
from typing import List, Dict, Any, Optional, Generator
import requests
from dataclasses import dataclass

from backend.utils.config import get_config


@dataclass
class OllamaResponse:
    """Represents an Ollama response"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    done: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "done": self.done,
        }


class OllamaClient:
    """
    Client for interacting with Ollama local LLMs
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Ollama client

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Default model to use
        """
        config = get_config().ollama

        self.base_url = base_url or config.base_url
        self.default_model = model or config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip('/')

        print(f"✅ Ollama client initialized")
        print(f"   URL: {self.base_url}")
        print(f"   Default model: {self.default_model}")

    def is_available(self) -> bool:
        """
        Check if Ollama server is available

        Returns:
            True if server is running
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]

            return models

        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """
        Generate text using Ollama

        Args:
            prompt: Input prompt
            model: Model to use (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: System message
            stream: Enable streaming

        Returns:
            OllamaResponse object
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Build request
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        try:
            if stream:
                return self._generate_stream(payload)
            else:
                return self._generate_sync(payload)

        except Exception as e:
            print(f"❌ Ollama generation error: {e}")
            return OllamaResponse(
                text=f"Error: {str(e)}",
                model=model,
                done=True
            )

    def _generate_sync(self, payload: Dict[str, Any]) -> OllamaResponse:
        """Generate text synchronously"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        data = response.json()

        return OllamaResponse(
            text=data.get('response', ''),
            model=payload['model'],
            prompt_tokens=data.get('prompt_eval_count', 0),
            completion_tokens=data.get('eval_count', 0),
            total_tokens=data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
            done=data.get('done', True)
        )

    def _generate_stream(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Generate text with streaming"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    yield data['response']

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """
        Chat with Ollama using message format

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            stream: Enable streaming

        Returns:
            OllamaResponse object
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            data = response.json()

            return OllamaResponse(
                text=data.get('message', {}).get('content', ''),
                model=model,
                prompt_tokens=data.get('prompt_eval_count', 0),
                completion_tokens=data.get('eval_count', 0),
                total_tokens=data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
                done=data.get('done', True)
            )

        except Exception as e:
            print(f"❌ Ollama chat error: {e}")
            return OllamaResponse(
                text=f"Error: {str(e)}",
                model=model,
                done=True
            )

    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            print(f"🔄 Pulling model: {model}")

            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600
            )
            response.raise_for_status()

            # Stream progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get('status', '')
                    if status:
                        print(f"   {status}")

            print(f"✅ Model pulled: {model}")
            return True

        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model

        Args:
            model: Model name

        Returns:
            Model information dictionary
        """
        model = model or self.default_model

        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model}
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {"error": str(e)}


# Convenience functions
def create_ollama_client(model: Optional[str] = None) -> OllamaClient:
    """Create Ollama client"""
    return OllamaClient(model=model)


def quick_generate(prompt: str, model: Optional[str] = None) -> str:
    """
    Quick text generation

    Args:
        prompt: Input prompt
        model: Model to use

    Returns:
        Generated text
    """
    client = OllamaClient(model=model)
    response = client.generate(prompt)
    return response.text


if __name__ == "__main__":
    # Test Ollama client
    print("Ollama Client Test")
    print("=" * 50)

    client = OllamaClient()

    # Check if available
    if client.is_available():
        print("✅ Ollama server is running")

        # List models
        models = client.list_models()
        print(f"\nAvailable models: {models}")

        if models:
            # Test generation
            print(f"\nTesting generation with {client.default_model}:")
            response = client.generate(
                "Explain what is machine learning in one sentence.",
                temperature=0.7
            )

            print(f"Response: {response.text}")
            print(f"Tokens: {response.total_tokens}")
        else:
            print("⚠️ No models available. Pull a model first:")
            print("   ollama pull llama3.3")
    else:
        print("❌ Ollama server not running")
        print("   Start with: ollama serve")

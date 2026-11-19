"""
Ollama LLM Client for Shrimp Interpretation System.
Handles communication with local Ollama server with automatic vision/text model selection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import requests


VISION_MODEL_PREFIXES = (
    "llava",
    "llama3.2-vision",
    "bakllava",
    "moondream",
    "minicpm-v",
    "qwen2.5vl",
    "qwen2-vl",
    "gemma3",
    "pixtral",
)


def _is_vision_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(prefix in name for prefix in VISION_MODEL_PREFIXES)


def _fmt(val: Any, default: str = "N/A") -> str:
    if val is None or val == "":
        return default
    return str(val)


class OllamaClient:
    """Client for interacting with local Ollama LLM server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.preferred_model = model or os.getenv("OLLAMA_MODEL")
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.last_error: Optional[str] = None

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.last_error = f"Cannot connect to Ollama server at {self.base_url}: {e}"
            return False

    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            result = response.json()
            return [m.get("name") for m in result.get("models", []) if m.get("name")]
        except Exception as e:
            self.last_error = f"Failed to list Ollama models: {e}"
            return []

    def resolve_model(self) -> Tuple[Optional[str], bool]:
        """
        Pick the best available Ollama model.
        Priority:
          1. Explicitly requested/configured model if installed
          2. Any installed vision-capable model (llava, llama3.2-vision, etc.)
          3. First installed text LLM (e.g. llama3.1:8b) for metrics-based analysis

        Returns:
            (model_name, supports_vision)
        """
        available = self.list_models()
        if not available:
            self.last_error = "No models installed in Ollama. Run `ollama pull llava:13b` or `ollama pull llama3.1:8b`."
            return None, False

        if self.preferred_model:
            # Exact match or prefix match (e.g. "llava" matches "llava:13b")
            for m in available:
                if m == self.preferred_model or m.startswith(f"{self.preferred_model}:"):
                    return m, _is_vision_model(m)

        # Prefer vision models when available
        for m in available:
            if _is_vision_model(m):
                return m, True

        # Fallback to first available text model
        return available[0], False

    def generate_interpretation(
        self,
        image_base64: Optional[str],
        metrics: Dict[str, Any],
        history: str,
        prompt_template: str,
        use_vision: Optional[bool] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """Generate shrimp interpretation using Ollama (vision or metrics-based)."""
        self.last_error = None
        model_name = model_override
        supports_vision = use_vision

        if not model_name or supports_vision is None:
            resolved_model, resolved_vision = self.resolve_model()
            if not resolved_model:
                return None
            model_name = model_name or resolved_model
            if supports_vision is None:
                supports_vision = resolved_vision

        try:
            prompt = prompt_template.format(
                count=_fmt(metrics.get("shrimp_count")),
                length=_fmt(metrics.get("avg_length_cm")),
                min_length=_fmt(metrics.get("min_length_cm")),
                max_length=_fmt(metrics.get("max_length_cm")),
                uniformity=_fmt(metrics.get("uniformity_pct")),
                weight=_fmt(metrics.get("avg_weight_g")),
                biomass=_fmt(metrics.get("estimated_biomass_g")),
                confidence=_fmt(metrics.get("avg_confidence_pct")),
                depth=_fmt(metrics.get("water_depth_m")),
                pond_id=_fmt(metrics.get("pond_id")),
                cycle_no=_fmt(metrics.get("cycle_no")),
                trend=_fmt(history),
            )

            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "15m",
                "options": {
                    "num_predict": 220,
                    "temperature": 0.3,
                },
            }
            if supports_vision and image_base64:
                payload["images"] = [image_base64]

            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            result = response.json()
            interpretation = result.get("response", "").strip()
            if not interpretation:
                self.last_error = f"Ollama model '{model_name}' returned an empty response."
                return None
            return interpretation

        except requests.exceptions.ConnectionError:
            self.last_error = f"Cannot connect to Ollama server at {self.base_url}. Ensure `ollama serve` is running."
            print(f"[Ollama] {self.last_error}")
            return None
        except requests.exceptions.Timeout:
            self.last_error = f"Ollama request timed out while waiting for model '{model_name}'."
            print(f"[Ollama] {self.last_error}")
            return None
        except Exception as e:
            self.last_error = f"Ollama generation failed ({model_name}): {e}"
            print(f"[Ollama] {self.last_error}")
            return None


def generate_shrimp_interpretation(
    image_base64: Optional[str],
    metrics: Dict[str, Any],
    history: str,
    prompt: str,
    model: Optional[str] = None,
    metrics_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate shrimp interpretation using the best available Ollama model.

    Returns a dict with keys:
      - interpretation (Optional[str])
      - model_used (Optional[str])
      - mode ("vision" | "metrics" | None)
      - error (Optional[str])
    """
    client = OllamaClient(model=model)

    if not client.check_connection():
        return {
            "interpretation": None,
            "model_used": None,
            "mode": None,
            "error": client.last_error,
        }

    resolved_model, supports_vision = client.resolve_model()
    if not resolved_model:
        return {
            "interpretation": None,
            "model_used": None,
            "mode": None,
            "error": client.last_error,
        }

    chosen_prompt = prompt if supports_vision else (metrics_prompt or prompt)
    text = client.generate_interpretation(
        image_base64=image_base64 if supports_vision else None,
        metrics=metrics,
        history=history,
        prompt_template=chosen_prompt,
        use_vision=supports_vision,
        model_override=resolved_model,
    )

    return {
        "interpretation": text,
        "model_used": resolved_model,
        "mode": "vision" if supports_vision else "metrics",
        "error": client.last_error if not text else None,
    }

"""Tests for TransformersProvider and steering extraction utilities."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from pathos.llm.transformers_provider import (
    TransformersProvider,
    find_ollama_gguf,
    _ollama_to_hf_id,
    _OLLAMA_TO_HF,
)


# ========== TestOllamaToHfId ==========

class TestOllamaToHfId:
    """Tests for Ollama → HuggingFace model ID mapping."""

    def test_exact_match(self) -> None:
        assert _ollama_to_hf_id("qwen3:4b") == "Qwen/Qwen3-4B"

    def test_llama_match(self) -> None:
        assert _ollama_to_hf_id("llama3.2:3b") == "meta-llama/Llama-3.2-3B-Instruct"

    def test_mistral_match(self) -> None:
        assert _ollama_to_hf_id("mistral:7b") == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_unknown_returns_none(self) -> None:
        assert _ollama_to_hf_id("nonexistent:99b") is None

    def test_base_name_fallback(self) -> None:
        """If exact tag doesn't match, try base name."""
        result = _ollama_to_hf_id("qwen3:unknown_tag")
        # Should find a qwen3:* entry
        assert result is not None
        assert "Qwen" in result

    def test_all_mappings_are_strings(self) -> None:
        for key, value in _OLLAMA_TO_HF.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert ":" in key  # All Ollama names have tag
            assert "/" in value  # All HF IDs have org/name


# ========== TestFindOllamaGguf ==========

class TestFindOllamaGguf:
    """Tests for find_ollama_gguf utility."""

    def test_not_found_no_ollama_dir(self) -> None:
        """No Ollama installation → returns None."""
        with patch.dict(os.environ, {"OLLAMA_MODELS": "/nonexistent/path"}):
            result = find_ollama_gguf("qwen3:4b")
            assert result is None

    def test_finds_model_in_manifest(self, tmp_path: Path) -> None:
        """Full test with mocked Ollama directory structure."""
        models_dir = tmp_path / "models"
        # Create manifest
        manifest_dir = models_dir / "manifests" / "registry.ollama.ai" / "library" / "testmodel" / "latest"
        manifest_dir.mkdir(parents=True)
        # Create blob
        blobs_dir = models_dir / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-abc123"
        blob_file.write_bytes(b"fake gguf data")
        # Write manifest pointing to blob
        manifest_data = {
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123"},
                {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:other"},
            ]
        }
        # The manifest IS the file at the tag level
        manifest_file = manifest_dir.parent / "latest"
        manifest_dir.rmdir()  # Remove dir, create as file
        manifest_file.write_text(json.dumps(manifest_data))

        with patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = find_ollama_gguf("testmodel:latest")
            assert result is not None
            assert result.name == "sha256-abc123"

    def test_default_tag_is_latest(self, tmp_path: Path) -> None:
        """Model name without tag defaults to 'latest'."""
        models_dir = tmp_path / "models"
        manifest_file = models_dir / "manifests" / "registry.ollama.ai" / "library" / "mymodel" / "latest"
        manifest_file.parent.mkdir(parents=True)
        blobs_dir = models_dir / "blobs"
        blobs_dir.mkdir(parents=True)
        blob = blobs_dir / "sha256-xyz"
        blob.write_bytes(b"data")
        manifest_file.write_text(json.dumps({
            "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:xyz"}]
        }))

        with patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = find_ollama_gguf("mymodel")  # No tag
            assert result is not None

    def test_missing_blob_returns_none(self, tmp_path: Path) -> None:
        """Manifest exists but blob file is missing."""
        models_dir = tmp_path / "models"
        manifest_file = models_dir / "manifests" / "registry.ollama.ai" / "library" / "broken" / "latest"
        manifest_file.parent.mkdir(parents=True)
        (models_dir / "blobs").mkdir(parents=True)
        manifest_file.write_text(json.dumps({
            "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:missing"}]
        }))

        with patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = find_ollama_gguf("broken:latest")
            assert result is None


# ========== TestTransformersProviderProperties ==========

class TestTransformersProviderProperties:
    """Tests for TransformersProvider properties (no model loading)."""

    def test_supports_steering(self) -> None:
        provider = TransformersProvider.__new__(TransformersProvider)
        provider._loaded = False
        provider._model = None
        assert provider.supports_steering is True

    def test_model_id(self) -> None:
        provider = TransformersProvider(model_id="qwen3:4b")
        assert provider.model == "qwen3:4b"

    def test_default_params(self) -> None:
        provider = TransformersProvider()
        assert provider._model_id == "qwen3:4b"
        assert provider._device_map == "auto"

    def test_custom_params(self) -> None:
        provider = TransformersProvider(
            model_id="llama3.2:3b",
            device_map="cpu",
            embed_model="nomic-embed-text",
        )
        assert provider._model_id == "llama3.2:3b"
        assert provider._device_map == "cpu"

    def test_steerable_model_triggers_load(self) -> None:
        """steerable_model calls _ensure_loaded."""
        provider = TransformersProvider.__new__(TransformersProvider)
        provider._loaded = True
        provider._model = MagicMock()
        assert provider.steerable_model is provider._model


# ========== TestTransformersProviderBuildPrompt ==========

class TestTransformersProviderBuildPrompt:
    """Tests for prompt building without loading a model."""

    def _make_provider_with_tokenizer(self, has_chat_template: bool = False) -> TransformersProvider:
        provider = TransformersProvider.__new__(TransformersProvider)
        provider._loaded = True
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        if has_chat_template:
            provider._tokenizer.apply_chat_template = MagicMock(
                return_value="<formatted prompt>"
            )
        else:
            del provider._tokenizer.apply_chat_template
        return provider

    def test_fallback_format(self) -> None:
        provider = self._make_provider_with_tokenizer(has_chat_template=False)
        result = provider._build_prompt(
            "You are helpful",
            [{"role": "user", "content": "Hello"}],
        )
        assert "<|system|>" in result
        assert "You are helpful" in result
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result

    def test_chat_template_used(self) -> None:
        provider = self._make_provider_with_tokenizer(has_chat_template=True)
        result = provider._build_prompt(
            "System prompt",
            [{"role": "user", "content": "Hi"}],
        )
        assert result == "<formatted prompt>"
        provider._tokenizer.apply_chat_template.assert_called_once()

    def test_multi_turn(self) -> None:
        provider = self._make_provider_with_tokenizer(has_chat_template=False)
        result = provider._build_prompt(
            "System",
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ],
        )
        assert "Q1" in result
        assert "A1" in result
        assert "Q2" in result


# ========== TestConfigIntegration ==========

class TestConfigIntegration:
    """Tests for config.py integration."""

    def test_transformers_provider_type_exists(self) -> None:
        from pathos.config import LLMProviderType
        assert hasattr(LLMProviderType, "TRANSFORMERS")
        assert LLMProviderType.TRANSFORMERS.value == "transformers"

    def test_settings_has_transformers_fields(self) -> None:
        from pathos.config import Settings
        s = Settings()
        assert hasattr(s, "transformers_model")
        assert hasattr(s, "transformers_device_map")
        assert s.transformers_model == "qwen3:4b"
        assert s.transformers_device_map == "auto"


# ========== TestSteeringExtractModule ==========

class TestSteeringExtractModule:
    """Tests for steering_extract utility (without actual model loading)."""

    def test_list_available_models_runs(self) -> None:
        """list_available_models doesn't crash even with no models."""
        from pathos.engine.steering_extract import list_available_models
        result = list_available_models()
        assert isinstance(result, list)

    def test_extract_import(self) -> None:
        """Module imports cleanly."""
        from pathos.engine.steering_extract import extract_and_cache, main
        assert callable(extract_and_cache)
        assert callable(main)

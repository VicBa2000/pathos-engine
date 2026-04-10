"""Configuracion del sistema."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProviderType(str, Enum):
    OLLAMA = "ollama"
    CLAUDE = "claude"


class Settings(BaseSettings):
    """Configuracion global de Pathos Engine."""

    # LLM Provider
    llm_provider: LLMProviderType = LLMProviderType.OLLAMA

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    ollama_embed_model: str = "nomic-embed-text"

    # Claude
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # -------------------------------------------------------------------------
    # Autonomous Research — Search Depth Configuration
    # -------------------------------------------------------------------------
    # These control how many articles the research loop processes per topic.
    # Each finding triggers a FULL emotional pipeline pass (1-2 LLM calls),
    # so increasing these values increases processing time proportionally.
    #
    # WARNING — Resource impact:
    #   Each finding = 1 full pipeline run (~2 LLM calls) + 1 self-inquiry (~1 LLM call)
    #   Example: search_results=10, process_top_n=5, subtopic_rounds=2, sub_process_n=3
    #   = 5 + (2 rounds * 3) = 11 findings = ~33 LLM calls per topic
    #
    #   Setting extreme values (e.g. 100 topics, 80 subtopics) will work but
    #   can take HOURS per topic and saturate your LLM. Start with defaults
    #   and increase gradually.
    #
    # Recommended ranges:
    #   search_results:   5-15  (how many DuckDuckGo results to fetch)
    #   process_top_n:    3-8   (how many of those to read and pipeline-process)
    #   subtopic_results: 3-10  (search results per subtopic deep-dive)
    #   sub_process_n:    2-5   (how many subtopic results to pipeline-process)
    #   subtopic_rounds:  1-3   (how many subtopic deep-dives per topic)
    #
    # Relationship between parameters:
    #   Total findings per topic = process_top_n + (subtopic_rounds * sub_process_n)
    #   Total LLM calls per topic ~ total_findings * 3 + 3 (picker + thinking + conclusion)
    #
    #   More findings = more emotional state accumulation = richer pipeline dynamics
    #   More subtopic_rounds = deeper exploration = stronger emotional "snowball"
    #   process_top_n and sub_process_n control breadth vs depth of each search
    #
    # The _normal variants are used for Normal/Lite modes.
    # The _raw variants are used for Raw/Extreme modes (deeper by default).
    # -------------------------------------------------------------------------

    # Normal / Lite mode depth
    research_search_results: int = Field(default=5, ge=1, le=50)
    research_process_top_n: int = Field(default=3, ge=1, le=20)
    research_subtopic_results: int = Field(default=3, ge=1, le=20)
    research_sub_process_n: int = Field(default=2, ge=1, le=10)
    research_subtopic_rounds: int = Field(default=1, ge=1, le=10)

    # Raw / Extreme mode depth (deeper by default for richer emotional buildup)
    research_raw_search_results: int = Field(default=10, ge=1, le=50)
    research_raw_process_top_n: int = Field(default=5, ge=1, le=20)
    research_raw_subtopic_results: int = Field(default=5, ge=1, le=20)
    research_raw_sub_process_n: int = Field(default=3, ge=1, le=10)
    research_raw_subtopic_rounds: int = Field(default=2, ge=1, le=10)

    model_config = {"env_prefix": "PATHOS_", "env_file": ".env"}

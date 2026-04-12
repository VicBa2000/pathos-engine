"""Emotional Conditioning Tokens — special token definitions and mappings.

Defines the set of special tokens used for emotional conditioning:
  <V+1> through <V+5> and <V-1> through <V-5> for valence
  <A+1> through <A+5> and <A-1> through <A-5> for arousal
  <D+1> through <D+5> and <D-1> through <D-5> for dominance
  <C+1> through <C+5> and <C-1> through <C-5> for certainty

Each level maps to a range of the continuous dimension:
  Level 1: |deviation| 0.0-0.2 (subtle)
  Level 2: |deviation| 0.2-0.4 (mild)
  Level 3: |deviation| 0.4-0.6 (moderate)
  Level 4: |deviation| 0.6-0.8 (strong)
  Level 5: |deviation| 0.8-1.0 (intense)

Usage:
  tokens = generate_emotional_tokens()
  token_str = state_to_tokens(valence=0.7, arousal=0.3, dominance=0.6, certainty=0.5)
  # → "<V+4><A-2><D+1><C+0>"
"""

from __future__ import annotations

from dataclasses import dataclass

# Emotional dimensions and their neutral points
DIMENSIONS = {
    "V": {"name": "valence", "neutral": 0.0, "range": (-1.0, 1.0)},
    "A": {"name": "arousal", "neutral": 0.5, "range": (0.0, 1.0)},
    "D": {"name": "dominance", "neutral": 0.5, "range": (0.0, 1.0)},
    "C": {"name": "certainty", "neutral": 0.5, "range": (0.0, 1.0)},
}

# Number of quantization levels per direction
NUM_LEVELS = 5


def generate_emotional_tokens() -> list[str]:
    """Generate the full set of emotional conditioning tokens.

    Returns list of special token strings to add to tokenizer.
    Includes a neutral token <EMO_NEUTRAL> and signed level tokens.
    """
    tokens: list[str] = ["<EMO_NEUTRAL>"]
    for dim in DIMENSIONS:
        for sign in ("+", "-"):
            for level in range(1, NUM_LEVELS + 1):
                tokens.append(f"<{dim}{sign}{level}>")
    return tokens


def _deviation_to_level(deviation: float) -> int:
    """Map absolute deviation from neutral to quantization level (0-5).

    Args:
        deviation: Absolute deviation from neutral [0, 1].

    Returns:
        Level 0-5 (0 = near neutral, 5 = extreme).
    """
    abs_dev = min(abs(deviation), 1.0)
    if abs_dev < 0.10:
        return 0
    level = int(abs_dev * NUM_LEVELS) + 1
    return min(level, NUM_LEVELS)


def state_to_tokens(
    valence: float,
    arousal: float,
    dominance: float,
    certainty: float,
    intensity: float = 1.0,
) -> str:
    """Convert continuous emotional state to conditioning token string.

    Args:
        valence: [-1, 1]
        arousal: [0, 1]
        dominance: [0, 1]
        certainty: [0, 1]
        intensity: Overall intensity [0, 1]. Below 0.05, returns neutral token.

    Returns:
        Token string like "<V+3><A-1><D+2><C+0>" or "<EMO_NEUTRAL>".
    """
    if intensity < 0.05:
        return "<EMO_NEUTRAL>"

    values = {"V": valence, "A": arousal, "D": dominance, "C": certainty}
    parts: list[str] = []

    for dim, val in values.items():
        neutral = DIMENSIONS[dim]["neutral"]
        deviation = val - neutral

        # For valence, deviation is already signed [-1, 1]
        # For arousal/dominance/certainty, deviation is [-0.5, 0.5] → scale to [-1, 1]
        if dim != "V":
            deviation *= 2.0

        level = _deviation_to_level(deviation)

        if level == 0:
            # Near neutral on this dimension — omit or use +0
            continue

        sign = "+" if deviation >= 0 else "-"
        parts.append(f"<{dim}{sign}{level}>")

    return "".join(parts) if parts else "<EMO_NEUTRAL>"


def tokens_to_state(token_str: str) -> dict[str, float]:
    """Parse conditioning token string back to approximate emotional state.

    Args:
        token_str: Token string like "<V+3><A-1>".

    Returns:
        Dict with dimension values (only dimensions present in tokens).
    """
    if token_str == "<EMO_NEUTRAL>" or not token_str:
        return {}

    import re
    pattern = r"<([VADC])([+-])(\d)>"
    matches = re.findall(pattern, token_str)

    state: dict[str, float] = {}
    for dim, sign, level_str in matches:
        level = int(level_str)
        deviation = (level / NUM_LEVELS)
        if sign == "-":
            deviation = -deviation

        neutral = DIMENSIONS[dim]["neutral"]
        if dim != "V":
            deviation /= 2.0  # scale back from [-1,1] to [-0.5, 0.5]

        state[DIMENSIONS[dim]["name"]] = round(neutral + deviation, 3)

    return state


@dataclass
class TokenConfig:
    """Configuration for emotional conditioning tokens."""

    tokens: list[str]
    num_dimensions: int = 4
    num_levels: int = NUM_LEVELS
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.total_tokens = len(self.tokens)


def get_token_config() -> TokenConfig:
    """Get the default token configuration."""
    tokens = generate_emotional_tokens()
    return TokenConfig(tokens=tokens)

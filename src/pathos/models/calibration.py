"""Calibration Mode - Modelos para calibracion emocional con humanos.

El usuario proporciona escenarios con su respuesta emocional esperada.
El sistema compara su output vs lo esperado y genera un perfil de calibracion.
"""

from pydantic import BaseModel, Field

from pathos.models.emotion import PrimaryEmotion


class CalibrationScenario(BaseModel):
    """Un escenario de calibracion enviado por el usuario.

    El usuario describe un estimulo y como esperaria sentirse.
    """

    stimulus: str
    expected_emotion: PrimaryEmotion
    expected_valence: float = Field(ge=-1, le=1)
    expected_arousal: float = Field(ge=0, le=1)
    expected_intensity: float = Field(ge=0, le=1)


class CalibrationResult(BaseModel):
    """Resultado de procesar un escenario: comparacion sistema vs humano."""

    scenario: CalibrationScenario

    # Lo que el sistema genero
    system_emotion: PrimaryEmotion
    system_valence: float
    system_arousal: float
    system_intensity: float

    # Deltas (esperado - sistema)
    valence_delta: float
    arousal_delta: float
    intensity_delta: float
    emotion_match: bool


class CalibrationProfile(BaseModel):
    """Perfil de calibracion derivado de los escenarios.

    Offsets y escalas que ajustan la generacion emocional para
    acercarse a las respuestas humanas del usuario.
    """

    valence_offset: float = Field(default=0.0, ge=-0.5, le=0.5)
    arousal_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    intensity_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    scenarios_used: int = Field(default=0, ge=0)
    emotion_accuracy: float = Field(default=0.0, ge=0, le=1)

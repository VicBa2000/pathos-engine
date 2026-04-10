"""Tests para Emotion Contagion - contagio emocional pre-cognitivo."""

import pytest

from pathos.engine.contagion import (
    compute_contagion_perturbation,
    detect_user_emotion,
    update_shadow_state,
)
from pathos.models.contagion import ShadowState, default_shadow_state
from pathos.models.personality import PersonalityProfile


class TestDetectUserEmotion:
    """Tests para deteccion de emocion del usuario."""

    def test_empty_input(self) -> None:
        v, a, s = detect_user_emotion("")
        assert s == 0.0

    def test_neutral_input(self) -> None:
        v, a, s = detect_user_emotion("Cual es la capital de Francia?")
        assert s == 0.0  # Sin senal emocional

    def test_positive_high_arousal(self) -> None:
        v, a, s = detect_user_emotion("Wow increible!! Me encanta esto!")
        assert v > 0.3
        assert a > 0.4
        assert s > 0.2

    def test_negative_high_arousal(self) -> None:
        v, a, s = detect_user_emotion("Esto es horrible, no funciona nada, estoy harto!")
        assert v < -0.3
        assert a > 0.4
        assert s > 0.2

    def test_negative_low_arousal(self) -> None:
        v, a, s = detect_user_emotion("Me siento triste y solo...")
        assert v < -0.3
        assert a < 0.5  # Baja energia
        assert s > 0.1

    def test_fear_anxiety(self) -> None:
        v, a, s = detect_user_emotion("Tengo miedo, estoy nervioso, no se que hacer")
        assert v < -0.2
        assert a > 0.5  # Miedo es alta activacion
        assert s > 0.2

    def test_caps_boost_arousal(self) -> None:
        _, a_normal, _ = detect_user_emotion("me encanta")
        _, a_caps, _ = detect_user_emotion("ME ENCANTA")
        # Caps deberia aumentar arousal
        assert a_caps >= a_normal

    def test_emoji_detection(self) -> None:
        v, _, s = detect_user_emotion("😢😭")
        assert v < 0
        assert s > 0

    def test_mixed_signals(self) -> None:
        v, a, s = detect_user_emotion("Gracias pero me siento triste")
        # Senal mixta — deberia detectar algo
        assert s > 0

    def test_exclamation_boost(self) -> None:
        _, a1, _ = detect_user_emotion("genial")
        _, a2, _ = detect_user_emotion("genial!!!")
        assert a2 >= a1

    def test_valence_range(self) -> None:
        """Valence siempre entre -1 y 1."""
        v, _, _ = detect_user_emotion("odio odio odio mierda horrible terrible")
        assert -1 <= v <= 1

    def test_arousal_range(self) -> None:
        """Arousal siempre entre 0 y 1."""
        _, a, _ = detect_user_emotion("PANICO MIEDO AAAH!!!")
        assert 0 <= a <= 1

    def test_signal_strength_range(self) -> None:
        """Signal strength siempre entre 0 y 1."""
        _, _, s = detect_user_emotion("wow increible genial fantastico amazing great love")
        assert 0 <= s <= 1

    def test_spanish_and_english(self) -> None:
        """Detecta emociones en ambos idiomas."""
        v_es, _, s_es = detect_user_emotion("me encanta, es genial")
        v_en, _, s_en = detect_user_emotion("I love it, it's great")
        assert v_es > 0 and s_es > 0
        assert v_en > 0 and s_en > 0


class TestUpdateShadowState:
    """Tests para actualizacion del shadow state."""

    def test_no_signal_decays(self) -> None:
        shadow = ShadowState(valence=0.5, arousal=0.6, signal_strength=0.5)
        updated = update_shadow_state(shadow, 0.0, 0.3, 0.0)
        assert abs(updated.valence) < abs(shadow.valence)
        assert updated.signal_strength < shadow.signal_strength

    def test_strong_signal_updates(self) -> None:
        shadow = default_shadow_state()
        updated = update_shadow_state(shadow, -0.7, 0.8, 0.6)
        assert updated.valence < 0  # Moved toward negative
        assert updated.arousal > shadow.arousal
        assert updated.signal_strength > 0

    def test_accumulated_contagion_grows(self) -> None:
        shadow = default_shadow_state()
        updated = update_shadow_state(shadow, 0.5, 0.6, 0.5)
        assert updated.accumulated_contagion > shadow.accumulated_contagion

    def test_accumulated_contagion_clamped(self) -> None:
        shadow = ShadowState(accumulated_contagion=0.95)
        updated = update_shadow_state(shadow, 0.5, 0.6, 0.8)
        assert updated.accumulated_contagion <= 1.0

    def test_turns_since_strong_resets(self) -> None:
        shadow = ShadowState(turns_since_strong_signal=5)
        updated = update_shadow_state(shadow, -0.5, 0.7, 0.5)
        assert updated.turns_since_strong_signal == 0

    def test_turns_since_strong_increments(self) -> None:
        shadow = ShadowState(turns_since_strong_signal=2)
        updated = update_shadow_state(shadow, 0.0, 0.3, 0.0)
        assert updated.turns_since_strong_signal == 3

    def test_shadow_valence_range(self) -> None:
        shadow = ShadowState(valence=-0.9)
        updated = update_shadow_state(shadow, -0.9, 0.8, 0.8)
        assert -1 <= updated.valence <= 1

    def test_shadow_arousal_range(self) -> None:
        shadow = ShadowState(arousal=0.9)
        updated = update_shadow_state(shadow, 0.5, 0.95, 0.8)
        assert 0 <= updated.arousal <= 1

    def test_inertia_preserves_old_state(self) -> None:
        """Shadow state no cambia drasticamente por un solo mensaje."""
        shadow = ShadowState(valence=0.5, arousal=0.5, signal_strength=0.3)
        updated = update_shadow_state(shadow, -0.8, 0.9, 0.3)
        # No deberia saltar de 0.5 a -0.8 de un turno
        assert updated.valence > -0.3


class TestComputeContagionPerturbation:
    """Tests para calculo de perturbacion de contagio."""

    def test_no_signal_no_perturbation(self) -> None:
        shadow = default_shadow_state()
        personality = PersonalityProfile()
        v, a = compute_contagion_perturbation(shadow, 0.0, 0.3, personality)
        assert v == 0.0
        assert a == 0.0

    def test_negative_signal_pulls_down(self) -> None:
        shadow = ShadowState(valence=-0.6, arousal=0.7, signal_strength=0.5)
        personality = PersonalityProfile(agreeableness=0.7)
        v, a = compute_contagion_perturbation(shadow, 0.0, 0.3, personality)
        assert v < 0  # Pulls valence down toward user's negative state

    def test_positive_signal_pulls_up(self) -> None:
        shadow = ShadowState(valence=0.7, arousal=0.6, signal_strength=0.5)
        personality = PersonalityProfile(agreeableness=0.7)
        v, a = compute_contagion_perturbation(shadow, 0.0, 0.3, personality)
        assert v > 0  # Pulls valence up toward user's positive state

    def test_high_agreeableness_more_susceptible(self) -> None:
        shadow = ShadowState(valence=-0.6, arousal=0.7, signal_strength=0.5)
        p_high = PersonalityProfile(agreeableness=0.9, emotional_reactivity=0.7)
        p_low = PersonalityProfile(agreeableness=0.2, emotional_reactivity=0.3)
        v_high, _ = compute_contagion_perturbation(shadow, 0.0, 0.3, p_high)
        v_low, _ = compute_contagion_perturbation(shadow, 0.0, 0.3, p_low)
        assert abs(v_high) > abs(v_low)

    def test_high_rapport_more_contagion(self) -> None:
        shadow = ShadowState(valence=-0.6, arousal=0.7, signal_strength=0.5)
        personality = PersonalityProfile(agreeableness=0.7)
        v_high_r, _ = compute_contagion_perturbation(shadow, 0.0, 0.3, personality, rapport=0.9)
        v_low_r, _ = compute_contagion_perturbation(shadow, 0.0, 0.3, personality, rapport=0.1)
        assert abs(v_high_r) > abs(v_low_r)

    def test_perturbation_clamped(self) -> None:
        shadow = ShadowState(valence=-1.0, arousal=1.0, signal_strength=1.0)
        personality = PersonalityProfile(agreeableness=1.0, emotional_reactivity=1.0, neuroticism=1.0)
        v, a = compute_contagion_perturbation(shadow, 1.0, 0.0, personality, rapport=1.0)
        assert -0.3 <= v <= 0.3
        assert -0.3 <= a <= 0.3

    def test_weak_signal_ignored(self) -> None:
        shadow = ShadowState(valence=-0.8, arousal=0.9, signal_strength=0.05)
        personality = PersonalityProfile(agreeableness=0.9)
        v, a = compute_contagion_perturbation(shadow, 0.0, 0.3, personality)
        assert v == 0.0
        assert a == 0.0


class TestContagionSusceptibility:
    """Tests para la propiedad contagion_susceptibility."""

    def test_default_personality(self) -> None:
        p = PersonalityProfile()
        s = p.contagion_susceptibility
        assert 0.1 <= s <= 1.0

    def test_empathic_profile_high(self) -> None:
        p = PersonalityProfile(agreeableness=0.9, emotional_reactivity=0.8, neuroticism=0.7)
        assert p.contagion_susceptibility > 0.6

    def test_resilient_profile_low(self) -> None:
        p = PersonalityProfile(agreeableness=0.2, emotional_reactivity=0.2, neuroticism=0.1, conscientiousness=0.8)
        assert p.contagion_susceptibility < 0.4

    def test_minimum_susceptibility(self) -> None:
        """Incluso el perfil menos susceptible tiene un minimo."""
        p = PersonalityProfile(agreeableness=0.0, emotional_reactivity=0.0, neuroticism=0.0, conscientiousness=1.0)
        assert p.contagion_susceptibility >= 0.1

    def test_susceptibility_range(self) -> None:
        """Siempre entre 0.1 y 1.0."""
        p = PersonalityProfile(agreeableness=1.0, emotional_reactivity=1.0, neuroticism=1.0, conscientiousness=0.0)
        assert 0.1 <= p.contagion_susceptibility <= 1.0


class TestDefaultShadowState:
    """Tests para el shadow state por defecto."""

    def test_neutral_defaults(self) -> None:
        s = default_shadow_state()
        assert s.valence == 0.0
        assert s.arousal == 0.3
        assert s.signal_strength == 0.0
        assert s.accumulated_contagion == 0.0
        assert s.turns_since_strong_signal == 0


class TestContagionIntegration:
    """Tests de integracion del pipeline de contagion."""

    def test_full_pipeline_positive(self) -> None:
        """Pipeline completo: deteccion -> shadow -> perturbacion positiva."""
        shadow = default_shadow_state()
        personality = PersonalityProfile(agreeableness=0.7)

        # 1. Detectar emocion positiva
        v, a, s = detect_user_emotion("Increible, me encanta!! Genial!")
        assert v > 0 and s > 0

        # 2. Actualizar shadow
        shadow = update_shadow_state(shadow, v, a, s)
        assert shadow.valence > 0

        # 3. Computar perturbacion
        pv, pa = compute_contagion_perturbation(shadow, 0.0, 0.3, personality, rapport=0.5)
        assert pv > 0  # Tira hacia positivo

    def test_full_pipeline_negative(self) -> None:
        """Pipeline completo: deteccion -> shadow -> perturbacion negativa."""
        shadow = default_shadow_state()
        personality = PersonalityProfile(agreeableness=0.7)

        # 1. Detectar emocion negativa
        v, a, s = detect_user_emotion("Estoy triste y solo, me siento mal...")
        assert v < 0 and s > 0

        # 2. Actualizar shadow
        shadow = update_shadow_state(shadow, v, a, s)
        assert shadow.valence < 0

        # 3. Computar perturbacion
        pv, pa = compute_contagion_perturbation(shadow, 0.0, 0.3, personality, rapport=0.5)
        assert pv < 0  # Tira hacia negativo

    def test_accumulation_over_turns(self) -> None:
        """El contagio se acumula con senales repetidas."""
        shadow = default_shadow_state()
        messages = [
            "Estoy triste",
            "Me siento mal",
            "No puedo mas, estoy deprimido",
        ]
        for msg in messages:
            v, a, s = detect_user_emotion(msg)
            shadow = update_shadow_state(shadow, v, a, s)

        assert shadow.accumulated_contagion > 0.1
        assert shadow.valence < -0.2

    def test_decay_without_signal(self) -> None:
        """Sin senal emocional, el shadow state decae."""
        shadow = ShadowState(valence=-0.6, arousal=0.7, signal_strength=0.5)
        neutral_messages = [
            "Cual es la capital de Francia?",
            "Dame la receta de paella.",
            "Que hora es?",
        ]
        for msg in neutral_messages:
            v, a, s = detect_user_emotion(msg)
            shadow = update_shadow_state(shadow, v, a, s)

        assert abs(shadow.valence) < abs(-0.6)  # Decayo
        assert shadow.signal_strength < 0.5  # Decayo

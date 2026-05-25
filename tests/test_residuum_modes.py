"""RESIDUUM × modes matrix (Section 8 of RESIDUUMREWORK.txt).

Verifies the per-mode behaviour the plan specifies:
  - F2 auto-enable policy (Lite OFF; Advanced/Raw/Extreme ON when Transformers
    path + library present; Ollama/no-capture OFF; manual toggle wins).
  - F4 mapping variant per mode (restricted / standard / expanded).
  - F4 steering fraction cap per mode (0.08 / 0.10 / 0.12 / 0.15).
  - F6 baseline compensation strength per mode (0.3 / 0.3 / 0.7 / 1.0).
"""

from __future__ import annotations

from pathos.engine.baseline_calibration import resolve_baseline_strength
from pathos.engine.steering import (
    resolve_stack_map_variant,
    resolve_steering_fraction_cap,
)
from pathos.main import _autoresolve_residuum
from pathos.state.manager import SessionState


# --- Fakes for the F2 auto-enable policy ---

class _CaptureProvider:
    """Transformers-like provider exposing the introspection hooks."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def has_capture(self) -> bool:
        return False

    def set_introspection(self, enabled: bool) -> None:
        self.calls.append(enabled)


class _OllamaProvider:
    """No residual hooks — cloud/Ollama path."""


class _FakeLib:
    pass


def _session(**flags) -> SessionState:
    s = SessionState()
    for k, v in flags.items():
        setattr(s, k, v)
    return s


class TestF2AutoEnablePerMode:
    def test_advanced_transformers_autoenables(self):
        s = _session()  # advanced default, auto default
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is True

    def test_raw_autoenables(self):
        s = _session(raw_mode=True)
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is True

    def test_extreme_autoenables(self):
        s = _session(extreme_mode=True)
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is True

    def test_lite_forces_off(self):
        s = _session(lite_mode=True)
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is False

    def test_ollama_provider_stays_off(self):
        s = _session()
        _autoresolve_residuum(s, _OllamaProvider(), _FakeLib())
        assert s.residuum.enabled is False

    def test_missing_library_stays_off(self):
        s = _session()
        _autoresolve_residuum(s, _CaptureProvider(), None)
        assert s.residuum.enabled is False

    def test_registers_hook_when_enabling(self):
        s = _session()
        prov = _CaptureProvider()
        _autoresolve_residuum(s, prov, _FakeLib())
        assert prov.calls == [True]

    def test_manual_off_choice_sticks(self):
        # User disabled via toggle -> residuum_auto False; auto must not re-enable.
        s = _session(residuum_auto=False)
        s.residuum.enabled = False
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is False

    def test_manual_on_choice_sticks(self):
        s = _session(residuum_auto=False)
        s.residuum.enabled = True
        _autoresolve_residuum(s, _CaptureProvider(), _FakeLib())
        assert s.residuum.enabled is True

    def test_auto_off_when_conditions_drop(self):
        # Was on under auto, then mode/provider no longer ready -> auto-off.
        s = _session()
        s.residuum.enabled = True
        _autoresolve_residuum(s, _OllamaProvider(), None)
        assert s.residuum.enabled is False


class TestF4MappingPerMode:
    def test_variants(self):
        assert resolve_stack_map_variant(lite_mode=True) == "restricted"
        assert resolve_stack_map_variant() == "standard"
        assert resolve_stack_map_variant(raw_mode=True) == "expanded"
        assert resolve_stack_map_variant(extreme_mode=True) == "expanded"


class TestF4CapPerMode:
    def test_caps(self):
        assert resolve_steering_fraction_cap(lite_mode=True) == 0.08
        assert resolve_steering_fraction_cap() == 0.10
        assert resolve_steering_fraction_cap(raw_mode=True) == 0.12
        assert resolve_steering_fraction_cap(extreme_mode=True) == 0.15


class TestF6StrengthPerMode:
    def test_strengths(self):
        assert resolve_baseline_strength(lite_mode=True) == 0.3
        assert resolve_baseline_strength() == 0.3
        assert resolve_baseline_strength(raw_mode=True) == 0.7
        assert resolve_baseline_strength(extreme_mode=True) == 1.0

    def test_extreme_precedence(self):
        assert resolve_baseline_strength(raw_mode=True, extreme_mode=True) == 1.0

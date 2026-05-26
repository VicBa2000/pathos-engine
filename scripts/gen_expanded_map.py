"""F4.5 — Generate stack_to_probe_map_expanded.json by MEASURING real probe cosines.

For each of the 19 Pathos stack emotions, start from the curated STANDARD
anchors and add same-cluster probes (the intense variants Raw/Extreme want:
enraged, spiteful, vindictive, grief-stricken, ...) ONLY when their cosine with
the anchor centroid is >= COHERENCE_THRESHOLD. This reproduces STANDARD's own
criterion (avoid vectors that cancel in the mean) so the expanded composite
stays coherent — richer/more intense steering without degrading the signal.

Run: PYTHONPATH=src python scripts/gen_expanded_map.py
Output: src/pathos/steering_data/stack_to_probe_map_expanded.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathos.engine.steering import ProbeLibrary, load_stack_to_probe_map

COHERENCE_THRESHOLD = 0.15  # same as STANDARD's intra-cluster criterion
MAX_PROBES_PER_EMOTION = 12  # keep composites bounded
DATA_DIR = Path("src/pathos/steering_data")


def main() -> None:
    lib = ProbeLibrary.load_family_from_cache("qwen3:4b", "single")
    if lib is None:
        raise SystemExit("qwen3:4b probe library not found")
    names = list(lib.emotion_names)
    clusters = list(lib.clusters)
    name_to_idx = {n: i for i, n in enumerate(names)}
    probes = lib.probes  # (171, hidden), unit-norm

    standard = load_stack_to_probe_map("standard")

    expanded: dict[str, list[str]] = {}
    notes: dict[str, str] = {}

    for emo, anchors in standard.items():
        if not anchors:  # mixed / neutral -> no steering, keep empty
            expanded[emo] = []
            continue
        anchor_idxs = [name_to_idx[a] for a in anchors if a in name_to_idx]
        if not anchor_idxs:
            expanded[emo] = list(anchors)
            continue
        # Anchor centroid direction (normalized).
        centroid = probes[anchor_idxs].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn < 1e-8:
            expanded[emo] = list(anchors)
            notes[emo] = "anchors cancel; kept as-is (no expansion)"
            continue
        centroid = centroid / cn
        anchor_clusters = {clusters[i] for i in anchor_idxs}

        # Candidate pool: same-cluster probes not already anchors.
        candidates = []
        for j, cl in enumerate(clusters):
            if cl in anchor_clusters and names[j] not in anchors:
                coh = float(probes[j] @ centroid)
                if coh >= COHERENCE_THRESHOLD:
                    candidates.append((names[j], coh))
        candidates.sort(key=lambda x: -x[1])
        added = [n for n, _ in candidates[: MAX_PROBES_PER_EMOTION - len(anchors)]]
        result = list(anchors) + added
        expanded[emo] = result
        notes[emo] = (
            f"{len(anchors)} anchors + {len(added)} coherent same-cluster probes "
            f"(cosine>= {COHERENCE_THRESHOLD} vs anchor centroid). "
            f"added: {', '.join(added) if added else '(none coherent)'}"
        )

    # Sanity: every expanded composite must be coherent (mean not near-zero).
    for emo, plist in expanded.items():
        if not plist:
            continue
        idxs = [name_to_idx[p] for p in plist if p in name_to_idx]
        m = probes[idxs].mean(axis=0)
        norm = float(np.linalg.norm(m))
        if norm < 0.10:
            print(f"WARNING: {emo} composite norm {norm:.3f} low (possible cancellation)")
        else:
            print(f"  {emo}: {len(plist)} probes, composite norm {norm:.3f}")

    out = {
        "version": 1,
        "variant": "expanded",
        "method": "sum_over_n",
        "description": (
            "F4.5 — EXPANDED mapping for Raw/Extreme. Each of the 19 Pathos stack "
            "emotions maps to the STANDARD anchors PLUS same-cluster intense "
            "variants, included ONLY when cosine vs the anchor centroid >= "
            f"{COHERENCE_THRESHOLD} (measured on qwen3_4b_171.npz). This reproduces "
            "STANDARD's anti-cancellation criterion so the richer composite stays "
            "coherent. Used by Raw (cap 0.12) and Extreme (cap 0.15); the cap still "
            "clamps the final magnitude, so EXPANDED enriches direction, not norm. "
            "NOT used by Lite/Advanced. Does not touch appraisal or the modulator "
            "bypass — only the probe set the steering composite draws from."
        ),
        "coherence_threshold": COHERENCE_THRESHOLD,
        "max_probes_per_emotion": MAX_PROBES_PER_EMOTION,
        "empty_means_no_steering": ["mixed", "neutral"],
        "mapping": expanded,
        "notes": notes,
    }
    out_path = DATA_DIR / "stack_to_probe_map_expanded.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

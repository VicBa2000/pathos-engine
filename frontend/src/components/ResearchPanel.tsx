import type { ResearchChatResponse } from "../types/emotion";
import "./ResearchPanel.css";

interface Props {
  data: ResearchChatResponse | null;
}

export function ResearchPanel({ data }: Props) {
  if (!data) {
    return (
      <div className="research-panel">
        <h2>Research Mode</h2>
        <div className="research-empty">
          Send a message to see all 10 systems in action...
        </div>
      </div>
    );
  }

  const { appraisal, homeostasis, memory_amplification, mood_congruence, emotion_generation,
    needs, social, regulation, reappraisal, temporal, meta_emotion, schemas, personality,
    contagion, somatic, creativity, immune, narrative, forecasting, voice, authenticity_metrics, emotional_state, emergent_emotions } = data;

  return (
    <div className="research-panel">
      <h2>Research Mode — Turn {data.turn_number}</h2>

      {/* Personality */}
      <Section title="Personality (Big Five)">
        <Row label="O/C/E/A/N" value={`${personality.openness.toFixed(1)}/${personality.conscientiousness.toFixed(1)}/${personality.extraversion.toFixed(1)}/${personality.agreeableness.toFixed(1)}/${personality.neuroticism.toFixed(1)}`} />
        <Row label="Variability" value={personality.variability.toFixed(3)} />
        <Row label="Reg. Capacity" value={personality.regulation_capacity_base.toFixed(3)} />
      </Section>

      {/* Appraisal */}
      <Section title="Appraisal (5D)">
        <Row label="Relevance" value={`nov=${appraisal.vector.relevance.novelty.toFixed(2)} sig=${appraisal.vector.relevance.personal_significance.toFixed(2)}`} />
        <Row label="Valence" value={`goal=${appraisal.vector.valence.goal_conduciveness.toFixed(2)} align=${appraisal.vector.valence.value_alignment.toFixed(2)}`} />
        <Row label="Coping" value={`ctrl=${appraisal.vector.coping.control.toFixed(2)} pwr=${appraisal.vector.coping.power.toFixed(2)}`} />
        <Row label="Agency" value={`agent=${appraisal.vector.agency.responsible_agent} fair=${appraisal.vector.agency.fairness.toFixed(2)}`} />
        <Row label="Computed" value={`V=${appraisal.computed_valence.toFixed(3)} A=${appraisal.computed_arousal.toFixed(3)} D=${appraisal.computed_dominance.toFixed(3)} C=${appraisal.computed_certainty.toFixed(3)}`} />
      </Section>

      {/* Homeostasis */}
      <Section title="Homeostasis">
        <Row label="Applied" value={homeostasis.applied ? "Yes" : "No (first turn)"} />
        {homeostasis.applied && (
          <>
            <Row label="Before" value={`V=${homeostasis.state_before.valence.toFixed(3)} A=${homeostasis.state_before.arousal.toFixed(3)} I=${homeostasis.state_before.intensity.toFixed(3)}`} />
            <Row label="After" value={`V=${homeostasis.state_after.valence.toFixed(3)} A=${homeostasis.state_after.arousal.toFixed(3)} I=${homeostasis.state_after.intensity.toFixed(3)}`} />
          </>
        )}
      </Section>

      {/* Temporal */}
      <Section title="Temporal Dynamics">
        <StatusRow label="Rumination" active={temporal.rumination_active}
          detail={temporal.rumination_active ? `${temporal.rumination_emotion} (${temporal.rumination_intensity.toFixed(2)})` : undefined} />
        <StatusRow label="Savoring" active={temporal.savoring_active}
          detail={temporal.savoring_active ? temporal.savoring_emotion ?? "" : undefined} />
        <StatusRow label="Anticipation" active={temporal.anticipation_active}
          detail={temporal.anticipation_active ? `${temporal.anticipation_emotion} (${temporal.anticipation_intensity.toFixed(2)})` : undefined} />
      </Section>

      {/* Amplification Sources */}
      <Section title="Amplification Sources">
        <Row label="Memory" value={`${(memory_amplification.amplification_factor * 100).toFixed(1)}% (${memory_amplification.memories_count} memories)`} />
        <Row label="Needs" value={`${(needs.amplification * 100).toFixed(1)}%`} />
        <Row label="Schema" value={schemas.primed_emotion ? `${schemas.primed_emotion} +${(schemas.priming_amplification * 100).toFixed(1)}%` : "none"} />
        <Row label="Social" value={`Vmod=${social.valence_modulation.toFixed(3)} Imod=${social.intensity_modulation.toFixed(3)}`} />
      </Section>

      {/* Needs */}
      <Section title="Computational Needs">
        <NeedBar label="Connection" value={needs.connection} />
        <NeedBar label="Competence" value={needs.competence} />
        <NeedBar label="Autonomy" value={needs.autonomy} />
        <NeedBar label="Coherence" value={needs.coherence} />
        <NeedBar label="Stimulation" value={needs.stimulation} />
        <NeedBar label="Safety" value={needs.safety} />
      </Section>

      {/* Social */}
      <Section title="Social Cognition">
        <Row label="Intent" value={`${social.perceived_intent >= 0 ? "+" : ""}${social.perceived_intent.toFixed(2)}`} />
        <Row label="Engagement" value={social.perceived_engagement.toFixed(2)} />
        <NeedBar label="Rapport" value={social.rapport} />
        <NeedBar label="Trust" value={social.trust_level} />
        <Row label="Style" value={social.communication_style} />
      </Section>

      {/* Emotion Contagion */}
      <Section title="Emotion Contagion">
        <Row label="Signal" value={contagion.signal_strength > 0.05
          ? `${contagion.signal_strength.toFixed(2)} (V=${contagion.detected_valence >= 0 ? "+" : ""}${contagion.detected_valence.toFixed(2)} A=${contagion.detected_arousal.toFixed(2)})`
          : "none"} />
        <Row label="Shadow" value={`V=${contagion.shadow_valence >= 0 ? "+" : ""}${contagion.shadow_valence.toFixed(3)} A=${contagion.shadow_arousal.toFixed(3)}`} />
        <Row label="Perturbation" value={contagion.contagion_perturbation_v !== 0 || contagion.contagion_perturbation_a !== 0
          ? `V=${contagion.contagion_perturbation_v >= 0 ? "+" : ""}${contagion.contagion_perturbation_v.toFixed(4)} A=${contagion.contagion_perturbation_a >= 0 ? "+" : ""}${contagion.contagion_perturbation_a.toFixed(4)}`
          : "none"} />
        <NeedBar label="Accumulated" value={contagion.accumulated_contagion} />
        <Row label="Susceptibility" value={contagion.susceptibility.toFixed(3)} />
      </Section>

      {/* Somatic Markers */}
      <Section title="Somatic Markers">
        <Row label="Markers" value={somatic.markers_count.toString()} />
        <Row label="Bias" value={somatic.somatic_bias !== 0
          ? `${somatic.somatic_bias >= 0 ? "+" : ""}${somatic.somatic_bias.toFixed(4)}`
          : "none"} />
        <Row label="Gut Feeling" value={somatic.gut_feeling ?? "none"} />
        <Row label="Pending" value={somatic.pending_category ?? "none"} />
      </Section>

      {/* Emotional Creativity */}
      <Section title="Emotional Creativity">
        <Row label="Thinking Mode" value={creativity.thinking_mode} />
        <NeedBar label="Creativity Level" value={creativity.creativity_level} />
        <Row label="Temp Modifier" value={creativity.temperature_modifier !== 0
          ? `${creativity.temperature_modifier >= 0 ? "+" : ""}${creativity.temperature_modifier.toFixed(3)}`
          : "none"} />
        {creativity.triggered_by.length > 0 && (
          <Row label="Triggered By" value={creativity.triggered_by.join(", ")} />
        )}
        {creativity.active_instructions.length > 0 && (
          <Row label="Instructions" value={`${creativity.active_instructions.length} active`} />
        )}
      </Section>

      {/* Emotional Immune System */}
      <Section title="Immune System">
        <Row label="Mode" value={immune.protection_mode} />
        {immune.protection_mode !== "none" && (
          <>
            <NeedBar label="Strength" value={immune.protection_strength} />
            <NeedBar label="Dampening" value={immune.reactivity_dampening} />
            <Row label="Streak" value={`${immune.negative_streak} turns`} />
            <Row label="Peak" value={immune.peak_negative_intensity.toFixed(3)} />
            {immune.recovery_turns > 0 && (
              <Row label="Recovery" value={`${immune.recovery_turns} turns`} />
            )}
            {immune.compartmentalized_topics.length > 0 && (
              <Row label="Compartment." value={immune.compartmentalized_topics.join(", ")} />
            )}
          </>
        )}
        <Row label="Activations" value={immune.total_activations.toString()} />
      </Section>

      {/* Narrative Self */}
      <Section title="Narrative Self">
        <Row label="Statements" value={narrative.identity_statements_count.toString()} />
        <NeedBar label="Coherence" value={narrative.coherence_score} />
        {narrative.top_statements.map((stmt, i) => (
          <Row key={i} label={`#${i + 1}`} value={stmt} />
        ))}
        {narrative.crisis_active && (
          <Row label="CRISIS" value={narrative.crisis_source || "active"} />
        )}
        {narrative.last_growth && (
          <Row label="Growth" value={narrative.last_growth} />
        )}
        <Row label="Stats" value={`+${narrative.total_reinforcements} / -${narrative.total_contradictions}`} />
      </Section>

      {/* Emotional Forecasting (optional) */}
      <Section title={`Forecasting ${forecasting.enabled ? "ON" : "OFF"}`}>
        {forecasting.enabled ? (
          <>
            <Row label="User Est." value={`V=${forecasting.user_valence.toFixed(3)} A=${forecasting.user_arousal.toFixed(3)} (${forecasting.user_dominant_signal})`} />
            <NeedBar label="Confidence" value={forecasting.user_confidence} />
            <Row label="Impact" value={forecasting.predicted_impact.toFixed(3)} />
            <Row label="Predicted" value={`V=${forecasting.predicted_user_valence.toFixed(3)} A=${forecasting.predicted_user_arousal.toFixed(3)}`} />
            {forecasting.risk_flag && (
              <Row label="RISK" value={forecasting.risk_reason} />
            )}
            {forecasting.recommendation && (
              <Row label="Rec." value={forecasting.recommendation} />
            )}
            <NeedBar label="Accuracy" value={forecasting.accuracy_score} />
            <Row label="Stats" value={`${forecasting.total_forecasts} forecasts, ${forecasting.total_evaluated} evaluated`} />
            {(forecasting.valence_bias !== 0 || forecasting.arousal_bias !== 0) && (
              <Row label="Bias" value={`V=${forecasting.valence_bias.toFixed(3)} A=${forecasting.arousal_bias.toFixed(3)}`} />
            )}
          </>
        ) : (
          <Row label="Status" value="Disabled — enable via toggle" />
        )}
      </Section>

      {/* Voice (optional) */}
      <Section title={`Voice ${voice.mode === "text_only" ? "OFF" : "ON"}`}>
        {voice.mode !== "text_only" ? (
          <>
            <Row label="Mode" value={voice.mode} />
            <Row label="Voice" value={voice.voice_key} />
            <Row label="Backend" value={voice.backend} />
            <Row label="Speed" value={voice.speed.toFixed(2)} />
            <Row label="Pitch" value={`${voice.pitch_semitones > 0 ? '+' : ''}${voice.pitch_semitones.toFixed(1)} st`} />
            <Row label="Volume" value={`${(voice.volume * 100).toFixed(0)}%`} />
            {voice.tremolo > 0 && <Row label="Tremolo" value={voice.tremolo.toFixed(3)} />}
            {voice.stage_direction && (
              <Row label="Direction" value={voice.stage_direction} />
            )}
            <Row label="TTS Audio" value={voice.audio_available ? "Available" : "—"} />
            <Row label="ASR" value={voice.asr_available ? "Active (Whisper)" : "Inactive"} />
            {voice.last_transcription && (
              <Row label="Last Transcript" value={voice.last_transcription} />
            )}
            {voice.detected_language && (
              <Row label="Detected Lang" value={voice.detected_language} />
            )}
          </>
        ) : (
          <Row label="Status" value="Disabled — enable via Voice toggle" />
        )}
      </Section>

      {/* Mood */}
      <Section title="Mood Congruence">
        <Row label="Label" value={mood_congruence.mood_label} />
        <Row label="Trend" value={mood_congruence.mood_trend} />
        <Row label="Bias" value={`V=${mood_congruence.valence_bias.toFixed(4)} A=${mood_congruence.arousal_bias.toFixed(4)}`} />
      </Section>

      {/* Emotion Generation */}
      <Section title="Emotion Generation (ODE)">
        <Row label="Raw" value={`V=${emotion_generation.raw_valence.toFixed(3)} A=${emotion_generation.raw_arousal.toFixed(3)} D=${emotion_generation.raw_dominance.toFixed(3)}`} />
        <Row label="Blended" value={`V=${emotion_generation.blended_valence.toFixed(3)} A=${emotion_generation.blended_arousal.toFixed(3)} D=${emotion_generation.blended_dominance.toFixed(3)}`} />
        <Row label="Intensity" value={`${emotion_generation.intensity_before_amplification.toFixed(3)} \u2192 ${emotion_generation.intensity_after_amplification.toFixed(3)}`} />
      </Section>

      {/* Emotional Stack */}
      <Section title="Emotional Stack (top 5)">
        {Object.entries(emotional_state.emotional_stack)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([emotion, activation]) => (
            <NeedBar key={emotion} label={emotion} value={activation} />
          ))}
      </Section>

      {/* Emergent Emotions */}
      {emergent_emotions.length > 0 && (
        <Section title="Emergent Emotions">
          <Row label="Active" value={emergent_emotions.join(", ")} />
        </Section>
      )}

      {/* Reappraisal */}
      <Section title="Reappraisal">
        <StatusRow label="Applied" active={reappraisal.applied}
          detail={reappraisal.applied
            ? `${reappraisal.strategy}: ${reappraisal.original_emotion} \u2192 ${reappraisal.reappraised_emotion} (\u0394V=${reappraisal.valence_change.toFixed(3)} \u0394I=${reappraisal.intensity_change.toFixed(3)})`
            : undefined} />
      </Section>

      {/* Regulation */}
      <Section title="Active Regulation">
        <Row label="Strategy" value={regulation.strategy_used ?? "none"} />
        <Row label="Capacity" value={`${(regulation.capacity_before * 100).toFixed(0)}% \u2192 ${(regulation.capacity_after * 100).toFixed(0)}%`} />
        <Row label="\u0394 Intensity" value={`-${regulation.intensity_reduced.toFixed(3)}`} />
        <Row label="Dissonance" value={regulation.suppression_dissonance.toFixed(3)} />
        {regulation.breakthrough && (
          <div className="research-breakthrough">EMOTIONAL BREAKTHROUGH</div>
        )}
      </Section>

      {/* Meta-Emotion */}
      <Section title="Meta-Emotion">
        <StatusRow label="Active" active={meta_emotion.active}
          detail={meta_emotion.active
            ? `${meta_emotion.meta_response} about ${meta_emotion.target_emotion} (${meta_emotion.intensity.toFixed(2)}): ${meta_emotion.reason}`
            : undefined} />
      </Section>

      {/* Schemas */}
      <Section title="Emotional Schemas">
        <Row label="Formed" value={String(schemas.schemas_count)} />
        <Row label="Pending" value={String(schemas.pending_patterns)} />
        {schemas.primed_emotion && (
          <Row label="Primed" value={`${schemas.primed_emotion} (+${(schemas.priming_amplification * 100).toFixed(1)}%)`} />
        )}
      </Section>

      {/* Metrics */}
      <Section title="Authenticity Metrics">
        <MetricBar label="Coherence" value={authenticity_metrics.coherence} />
        <MetricBar label="Continuity" value={authenticity_metrics.continuity} />
        <MetricBar label="Proportionality" value={authenticity_metrics.proportionality} />
        <MetricBar label="Recovery" value={authenticity_metrics.recovery} />
        <MetricBar label="Overall" value={authenticity_metrics.overall} highlight />
      </Section>

      {/* Behavior Prompt */}
      <Section title="Behavior Prompt">
        <pre className="research-prompt">{data.behavior_prompt}</pre>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="research-section">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="research-row">
      <span className="research-row__label">{label}</span>
      <span className="research-row__value">{value}</span>
    </div>
  );
}

function StatusRow({ label, active, detail }: { label: string; active: boolean; detail?: string }) {
  return (
    <div className="research-row">
      <span className="research-row__label">{label}</span>
      <span className={`research-status ${active ? "research-status--on" : "research-status--off"}`}>
        {active ? "ON" : "off"}
      </span>
      {detail && <span className="research-row__value">{detail}</span>}
    </div>
  );
}

function NeedBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(value * 100, 100);
  const color = value > 0.7 ? "#e67e22" : value > 0.4 ? "#6a6acc" : "#3a3a5a";
  return (
    <div className="research-metric">
      <span className="research-metric__label">{label}</span>
      <div className="research-metric__track">
        <div className="research-metric__fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="research-metric__value">{value.toFixed(2)}</span>
    </div>
  );
}

function MetricBar({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  const color = value > 0.7 ? "#2ecc71" : value > 0.4 ? "#f1c40f" : "#e74c3c";
  return (
    <div className="research-metric">
      <span className={`research-metric__label ${highlight ? "research-metric__label--highlight" : ""}`}>{label}</span>
      <div className="research-metric__track">
        <div className="research-metric__fill" style={{ width: `${value * 100}%`, backgroundColor: color }} />
      </div>
      <span className="research-metric__value">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

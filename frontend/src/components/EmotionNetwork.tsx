import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { EmotionalState, PrimaryEmotion } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import "./EmotionNetwork.css";

interface Props {
  state: EmotionalState;
  history: Array<{ emotion: PrimaryEmotion; intensity: number }>;
}

/** Node positions: emotions arranged on valence-arousal space as a neural map */
const NODES: Record<PrimaryEmotion, { x: number; y: number; group: string }> = {
  // Positive high energy (top-right)
  excitement:     { x: 260, y: 40,  group: "pos-hi" },
  joy:            { x: 225, y: 75,  group: "pos-hi" },
  hope:           { x: 195, y: 45,  group: "pos-hi" },
  gratitude:      { x: 205, y: 110, group: "pos-hi" },
  // Positive low energy (bottom-right)
  contentment:    { x: 245, y: 180, group: "pos-lo" },
  relief:         { x: 210, y: 205, group: "pos-lo" },
  // Negative high energy (top-left)
  anger:          { x: 30,  y: 40,  group: "neg-hi" },
  frustration:    { x: 60,  y: 75,  group: "neg-hi" },
  fear:           { x: 38,  y: 110, group: "neg-hi" },
  anxiety:        { x: 78,  y: 45,  group: "neg-hi" },
  // Negative low energy (bottom-left)
  sadness:        { x: 38,  y: 200, group: "neg-lo" },
  helplessness:   { x: 55,  y: 245, group: "neg-lo" },
  disappointment: { x: 85,  y: 220, group: "neg-lo" },
  // Neutral / ambiguous (center)
  surprise:       { x: 145, y: 30,  group: "neutral" },
  alertness:      { x: 115, y: 80,  group: "neutral" },
  contemplation:  { x: 160, y: 175, group: "neutral" },
  indifference:   { x: 125, y: 245, group: "neutral" },
  mixed:          { x: 145, y: 135, group: "neutral" },
  neutral:        { x: 145, y: 220, group: "neutral" },
};

const W = 290;
const H = 275;

export function EmotionNetwork({ state, history }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const stack = state.emotional_stack ?? {};
    const currentEmotion = state.primary_emotion;
    const currentColor = EMOTION_COLORS[currentEmotion] ?? "#6666aa";

    // Defs
    const defs = svg.append("defs");

    // Glow filter
    const glow = defs.append("filter").attr("id", "en-glow")
      .attr("x", "-80%").attr("y", "-80%").attr("width", "260%").attr("height", "260%");
    glow.append("feGaussianBlur").attr("stdDeviation", "5").attr("result", "blur");
    glow.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).join("feMergeNode").attr("in", d => d);

    const glowSoft = defs.append("filter").attr("id", "en-glow-soft")
      .attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    glowSoft.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "blur");
    glowSoft.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).join("feMergeNode").attr("in", d => d);

    // Background
    const bg = svg.append("g");

    // Crosshair
    bg.append("line").attr("x1", W / 2).attr("y1", 10).attr("x2", W / 2).attr("y2", H - 10)
      .attr("stroke", "#1a1a30").attr("stroke-width", 0.5).attr("stroke-dasharray", "3 3");
    bg.append("line").attr("x1", 10).attr("y1", H / 2).attr("x2", W - 10).attr("y2", H / 2)
      .attr("stroke", "#1a1a30").attr("stroke-width", 0.5).attr("stroke-dasharray", "3 3");

    // Transition trails
    const trailG = svg.append("g");
    if (history.length > 1) {
      for (let i = 1; i < history.length; i++) {
        const prev = history[i - 1];
        const curr = history[i];
        const pNode = NODES[prev.emotion];
        const cNode = NODES[curr.emotion];
        if (!pNode || !cNode || prev.emotion === curr.emotion) continue;

        const age = (history.length - i) / history.length;
        const opacity = Math.max(0.06, 0.35 * (1 - age));
        const color = EMOTION_COLORS[curr.emotion] ?? "#4a4a8a";

        const mx = (pNode.x + cNode.x) / 2;
        const my = (pNode.y + cNode.y) / 2 - 15;
        trailG.append("path")
          .attr("d", `M${pNode.x},${pNode.y} Q${mx},${my} ${cNode.x},${cNode.y}`)
          .attr("fill", "none")
          .attr("stroke", color)
          .attr("stroke-width", 1 + curr.intensity * 1.5)
          .attr("opacity", opacity)
          .attr("stroke-linecap", "round");
      }

      // Animated particle on latest transition
      if (history.length >= 2) {
        const prev = history[history.length - 2];
        const curr = history[history.length - 1];
        const pNode = NODES[prev.emotion];
        const cNode = NODES[curr.emotion];
        if (pNode && cNode && prev.emotion !== curr.emotion) {
          const particle = trailG.append("circle")
            .attr("cx", pNode.x).attr("cy", pNode.y)
            .attr("r", 3)
            .attr("fill", currentColor)
            .attr("filter", "url(#en-glow)")
            .attr("opacity", 0.9);

          particle.transition()
            .duration(800)
            .ease(d3.easeCubicInOut)
            .attr("cx", cNode.x).attr("cy", cNode.y)
            .attr("r", 5)
            .transition()
            .duration(400)
            .attr("r", 2)
            .attr("opacity", 0.3);
        }
      }
    }

    // Emotion nodes
    const nodesG = svg.append("g");

    for (const [emotion, pos] of Object.entries(NODES)) {
      const emKey = emotion as PrimaryEmotion;
      const isCurrent = emKey === currentEmotion;
      const activation = stack[emotion] ?? 0;
      const color = EMOTION_COLORS[emKey] ?? "#6666aa";

      const g = nodesG.append("g")
        .attr("transform", `translate(${pos.x},${pos.y})`);

      const baseR = isCurrent
        ? 8 + state.intensity * 10
        : 3 + activation * 11;

      // Outer glow
      if (activation > 0.08 || isCurrent) {
        g.append("circle")
          .attr("r", baseR + 6)
          .attr("fill", color)
          .attr("opacity", isCurrent ? 0.15 : activation * 0.12)
          .attr("filter", isCurrent ? "url(#en-glow)" : "url(#en-glow-soft)");
      }

      // Node
      g.append("circle")
        .attr("r", baseR)
        .attr("fill", isCurrent ? color : (activation > 0.05 ? color : "#1a1a30"))
        .attr("stroke", activation > 0.05 ? color : "#222240")
        .attr("stroke-width", isCurrent ? 1.5 : (activation > 0.05 ? 0.8 : 0.4))
        .attr("opacity", isCurrent ? 1 : Math.max(0.2, activation * 2.5));

      // Label
      const labelOpacity = isCurrent ? 1 : Math.max(0.2, activation * 3);
      g.append("text")
        .attr("y", baseR + 10)
        .attr("text-anchor", "middle")
        .attr("font-size", isCurrent ? "8px" : "6px")
        .attr("font-weight", isCurrent ? "700" : "400")
        .attr("fill", isCurrent ? color : "#556")
        .attr("opacity", labelOpacity)
        .text(emotion);

      // Activation %
      if ((activation > 0.1 || isCurrent) && activation > 0) {
        g.append("text")
          .attr("y", -baseR - 4)
          .attr("text-anchor", "middle")
          .attr("font-size", "6px")
          .attr("font-family", "monospace")
          .attr("fill", color)
          .attr("opacity", 0.7)
          .text(`${(activation * 100).toFixed(0)}%`);
      }
    }

    // Transition label
    if (history.length >= 2) {
      const prev = history[history.length - 2];
      const curr = history[history.length - 1];
      if (prev.emotion !== curr.emotion) {
        svg.append("text")
          .attr("x", W / 2)
          .attr("y", H - 4)
          .attr("text-anchor", "middle")
          .attr("font-size", "8px")
          .attr("font-family", "monospace")
          .attr("fill", currentColor)
          .attr("font-weight", "600")
          .text(`${prev.emotion} \u2192 ${curr.emotion}`)
          .attr("opacity", 0)
          .transition().duration(500).attr("opacity", 0.8);
      }
    }

  }, [state, history]);

  return (
    <div className="network">
      <div className="network__header">
        <span className="network__title">Network</span>
        <span className="network__current">
          {state.primary_emotion} {(state.intensity * 100).toFixed(0)}%
        </span>
      </div>
      <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`} />
    </div>
  );
}

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { EmotionalState } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";

interface Props {
  state: EmotionalState;
  history: Array<{ valence: number; arousal: number; emotion: string }>;
}

const SIZE = 200;
const MARGIN = 25;
const R = (SIZE - MARGIN * 2) / 2;

/** Valence-Arousal circumplex chart (Russell model). */
export function CircumplexChart({ state, history }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const cx = SIZE / 2;
    const cy = SIZE / 2;

    // Background circles
    const g = svg.append("g");
    for (const r of [0.25, 0.5, 0.75, 1]) {
      g.append("circle")
        .attr("cx", cx).attr("cy", cy)
        .attr("r", R * r)
        .attr("fill", "none")
        .attr("stroke", "#2a2a4a")
        .attr("stroke-width", 0.5);
    }

    // Axes
    g.append("line").attr("x1", MARGIN).attr("y1", cy).attr("x2", SIZE - MARGIN).attr("y2", cy)
      .attr("stroke", "#3a3a5a").attr("stroke-width", 0.5);
    g.append("line").attr("x1", cx).attr("y1", MARGIN).attr("x2", cx).attr("y2", SIZE - MARGIN)
      .attr("stroke", "#3a3a5a").attr("stroke-width", 0.5);

    // Labels
    g.append("text").attr("x", SIZE - MARGIN + 5).attr("y", cy + 4).text("Pos").attr("text-anchor", "start").attr("fill", "#6666aa").attr("font-size", "10px");
    g.append("text").attr("x", MARGIN - 5).attr("y", cy + 4).text("Neg").attr("text-anchor", "end").attr("fill", "#6666aa").attr("font-size", "10px");
    g.append("text").attr("x", cx).attr("y", MARGIN - 8).text("High").attr("fill", "#6666aa").attr("font-size", "10px").attr("text-anchor", "middle");
    g.append("text").attr("x", cx).attr("y", SIZE - MARGIN + 15).text("Low").attr("fill", "#6666aa").attr("font-size", "10px").attr("text-anchor", "middle");

    // Quadrant labels (faint)
    const ql = { fill: "#333355", "font-size": "9px", "text-anchor": "middle" as const };
    g.append("text").attr("x", cx + R * 0.5).attr("y", cy - R * 0.5).text("Excited").attr("fill", ql.fill).attr("font-size", ql["font-size"]).attr("text-anchor", ql["text-anchor"]);
    g.append("text").attr("x", cx - R * 0.5).attr("y", cy - R * 0.5).text("Tense").attr("fill", ql.fill).attr("font-size", ql["font-size"]).attr("text-anchor", ql["text-anchor"]);
    g.append("text").attr("x", cx + R * 0.5).attr("y", cy + R * 0.5).text("Calm").attr("fill", ql.fill).attr("font-size", ql["font-size"]).attr("text-anchor", ql["text-anchor"]);
    g.append("text").attr("x", cx - R * 0.5).attr("y", cy + R * 0.5).text("Sad").attr("fill", ql.fill).attr("font-size", ql["font-size"]).attr("text-anchor", ql["text-anchor"]);

    // Map valence (-1..1) -> x, arousal (0..1) -> y (inverted)
    const xScale = (v: number) => cx + v * R;
    const yScale = (a: number) => cy - (a * 2 - 1) * R; // 0->bottom, 1->top

    // Trail (history)
    if (history.length > 1) {
      const line = d3.line<{ valence: number; arousal: number }>()
        .x(d => xScale(d.valence))
        .y(d => yScale(d.arousal))
        .curve(d3.curveCatmullRom.alpha(0.5));

      g.append("path")
        .datum(history)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "#4a4a8a")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4 2")
        .attr("opacity", 0.6);

      // History dots
      history.forEach((h, i) => {
        if (i === history.length - 1) return; // skip current
        g.append("circle")
          .attr("cx", xScale(h.valence))
          .attr("cy", yScale(h.arousal))
          .attr("r", 3)
          .attr("fill", EMOTION_COLORS[h.emotion as keyof typeof EMOTION_COLORS] ?? "#6666aa")
          .attr("opacity", 0.4 + (i / history.length) * 0.4);
      });
    }

    // Current point (pulsing)
    const color = EMOTION_COLORS[state.primary_emotion] ?? "#fff";
    const px = xScale(state.valence);
    const py = yScale(state.arousal);

    // Glow
    g.append("circle")
      .attr("cx", px).attr("cy", py)
      .attr("r", 8 + state.intensity * 12)
      .attr("fill", color)
      .attr("opacity", 0.15);

    // Point
    g.append("circle")
      .attr("cx", px).attr("cy", py)
      .attr("r", 5 + state.intensity * 4)
      .attr("fill", color)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5);

  }, [state, history]);

  return (
    <div className="circumplex-chart">
      <h3>Valence-Arousal Space</h3>
      <svg ref={svgRef} width={SIZE} height={SIZE} />
    </div>
  );
}

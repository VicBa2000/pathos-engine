import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { PrimaryEmotion } from "../types/emotion";
import { EMOTION_COLORS } from "../types/emotion";
import "./JourneyTimeline.css";

interface JourneyPoint {
  turn: number;
  emotion: PrimaryEmotion;
  valence: number;
  arousal: number;
  intensity: number;
}

interface Props {
  journey: JourneyPoint[];
}

const HEIGHT = 120;
const MARGIN = { top: 20, right: 20, bottom: 25, left: 40 };

export function JourneyTimeline({ journey }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || journey.length === 0) return;

    const width = containerRef.current.clientWidth;
    const svg = d3.select(svgRef.current);
    svg.attr("width", width);
    svg.selectAll("*").remove();

    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const xScale = d3.scaleLinear()
      .domain([1, Math.max(journey.length, 2)])
      .range([0, innerW]);

    const yScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([innerH, 0]);

    // Zero line
    g.append("line")
      .attr("x1", 0).attr("y1", yScale(0))
      .attr("x2", innerW).attr("y2", yScale(0))
      .attr("stroke", "#3a3a5a").attr("stroke-width", 0.5).attr("stroke-dasharray", "4 2");

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(Math.min(journey.length, 10)).tickFormat(d => `T${d}`))
      .attr("color", "#555577").attr("font-size", "10px");

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-innerW))
      .attr("color", "#333355").attr("font-size", "10px")
      .selectAll("line").attr("stroke", "#222244");

    // Area
    const area = d3.area<JourneyPoint>()
      .x(d => xScale(d.turn))
      .y0(yScale(0))
      .y1(d => yScale(d.valence))
      .curve(d3.curveCatmullRom.alpha(0.5));

    g.append("path")
      .datum(journey)
      .attr("d", area)
      .attr("fill", "url(#valenceGradient)")
      .attr("opacity", 0.3);

    // Gradient
    const defs = svg.append("defs");
    const gradient = defs.append("linearGradient").attr("id", "valenceGradient").attr("x1", "0").attr("y1", "0").attr("x2", "0").attr("y2", "1");
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "#2ecc71");
    gradient.append("stop").attr("offset", "50%").attr("stop-color", "#333");
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "#e74c3c");

    // Line
    const line = d3.line<JourneyPoint>()
      .x(d => xScale(d.turn))
      .y(d => yScale(d.valence))
      .curve(d3.curveCatmullRom.alpha(0.5));

    g.append("path")
      .datum(journey)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#8888cc")
      .attr("stroke-width", 2);

    // Dots
    journey.forEach(p => {
      const color = EMOTION_COLORS[p.emotion];
      g.append("circle")
        .attr("cx", xScale(p.turn))
        .attr("cy", yScale(p.valence))
        .attr("r", 4 + p.intensity * 4)
        .attr("fill", color)
        .attr("stroke", "#fff")
        .attr("stroke-width", 1);

      g.append("text")
        .attr("x", xScale(p.turn))
        .attr("y", yScale(p.valence) - 10 - p.intensity * 4)
        .attr("text-anchor", "middle")
        .attr("fill", color)
        .attr("font-size", "9px")
        .text(p.emotion);
    });

  }, [journey]);

  return (
    <div className="journey-timeline" ref={containerRef}>
      <h3>Emotional Journey</h3>
      {journey.length === 0 ? (
        <div className="journey-empty">Journey will appear after the first message...</div>
      ) : (
        <svg ref={svgRef} height={HEIGHT} />
      )}
    </div>
  );
}

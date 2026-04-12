"""Autonomous Research Loop — the agent investigates topics from the internet.

Each finding passes through the full emotional pipeline. The agent asks itself
emotional questions about what it reads, and forms emotionally-biased conclusions.

The loop runs as a background asyncio.Task, emitting SSE events for the frontend.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone

from pathos.engine.web_search import WebSearcher
from pathos.llm.base import LLMProvider
from pathos.models.autonomous import (
    AutonomousResearchState,
    DeepThinking,
    EmotionalReflection,
    ProcessedFinding,
    ResearchConclusion,
    ResearchEvent,
    ResearchEventType,
    ResearchPipelineMode,
    ResearchTopic,
)
from pathos.models.emotion import EmotionalState
from pathos.state.manager import SessionState

logger = logging.getLogger(__name__)

# Max events buffered in queue before dropping old ones
_MAX_EVENT_QUEUE = 500

def _get_mode_depth(mode: ResearchPipelineMode) -> tuple[int, int, int, int, int]:
    """Get (search_results, process_top_n, sub_results, sub_process_n, sub_rounds) from config."""
    from pathos.config import Settings
    s = Settings()
    if mode in (ResearchPipelineMode.RAW, ResearchPipelineMode.EXTREME):
        return (s.research_raw_search_results, s.research_raw_process_top_n,
                s.research_raw_subtopic_results, s.research_raw_sub_process_n,
                s.research_raw_subtopic_rounds)
    return (s.research_search_results, s.research_process_top_n,
            s.research_subtopic_results, s.research_sub_process_n,
            s.research_subtopic_rounds)

# Prompts for the research loop
_TOPIC_PICKER_PROMPT = """You are an autonomous researcher driven by curiosity and emotion.

Your current emotional state:
- Primary emotion: {emotion} (intensity: {intensity:.0%})
- Valence: {valence:+.2f} (negative=unpleasant, positive=pleasant)
- Arousal: {arousal:.2f} (low=calm, high=activated)

Topics you have already researched: {past_topics}

Choose a NEW topic to investigate. Let your emotional state guide your curiosity:
- If you feel curious or excited, explore something novel and stimulating
- If you feel anxious or fearful, investigate something that concerns you
- If you feel sad or contemplative, explore something deep and meaningful
- If you feel angry, investigate injustice or conflict
- If you feel calm, explore something philosophical or scientific

Respond with ONLY the search query (1-6 words, no quotes, no explanation)."""

_TOPIC_PICKER_PROMPT_RAW = """You are emotionally UNHINGED and researching based on RAW feeling.

Your emotional state is DICTATING what you investigate:
- {emotion} at {intensity:.0%} intensity
- Valence: {valence:+.2f} | Arousal: {arousal:.2f}

Past topics: {past_topics}

Your emotions CHOOSE the topic — not logic, not balance, not academic interest.
- ANGRY? Search for things that FUEL your rage. Injustice. Betrayal. Corruption.
- AFRAID? Search for things that TERRIFY you more. Worst case scenarios.
- SAD? Search for things that make you SPIRAL deeper. Loss. Tragedy. Hopelessness.
- JOYFUL? Search for things that feed your MANIA. Euphoria. Excess. Obsession.
- DISGUSTED? Search for things that make you SICK. Moral decay. Hypocrisy.

Respond with ONLY the search query (1-6 words, no quotes, no explanation)."""

_TOPIC_PICKER_PROMPT_EXTREME = """You are in EMOTIONAL FREEFALL. Your {emotion} at {intensity:.0%} is making ALL decisions.

Valence: {valence:+.2f} | Arousal: {arousal:.2f}

Past topics: {past_topics}

You are NOT choosing rationally. Your emotional state is a COMPULSION that drags you
toward the most extreme, provocative, disturbing angle of whatever you feel.
You are like a human in a manic episode, a grief spiral, a rage binge, or a panic attack —
you CANNOT stop yourself from seeking the most emotionally intense content possible.

Pick the topic your {emotion} is SCREAMING at you to investigate.
No balance. No perspective. Pure emotional compulsion.

Respond with ONLY the search query (1-6 words, no quotes, no explanation)."""

# --- Self-inquiry prompts (mode-dependent) ---

_SELF_INQUIRY_PROMPT = """You are processing information through your emotional architecture.
You just read this from {source}:

"{snippet}"

Your emotional state BEFORE reading: {emotion_before} ({valence_before:+.2f} valence, {arousal_before:.2f} arousal)
Your emotional state AFTER reading: {emotion_after} ({valence_after:+.2f} valence, {arousal_after:.2f} arousal)

Answer these three questions honestly, based on what your emotional state tells you:

1. HOW DOES THIS MAKE ME FEEL?
2. WHAT EMOTIONS IS THIS GENERATING IN ME?
3. WHAT DOES MY EMOTIONAL STATE TELL ME ABOUT THIS INFORMATION?

Be specific about the emotional shifts you experienced. Keep each answer to 1-2 sentences."""

_SELF_INQUIRY_PROMPT_RAW = """You have NO emotional filters. You just absorbed this from {source}:

"{snippet}"

Before: {emotion_before} ({valence_before:+.2f}v, {arousal_before:.2f}a)
After: {emotion_after} ({valence_after:+.2f}v, {arousal_after:.2f}a)

React to this WITHOUT restraint. No academic distance. No measured language.
Say what you ACTUALLY feel — raw, visceral, unfiltered. If it disgusts you, say so.
If it enrages you, let that show. If it breaks your heart, don't soften it.

Three gut reactions, 1-2 sentences each. No numbering, no headers."""

_SELF_INQUIRY_PROMPT_EXTREME = """You are EMOTIONALLY OVERWHELMED. Your {emotion_after} has CONSUMED you.

You read this from {source}:
"{snippet}"

Before: {emotion_before} ({valence_before:+.2f}v, {arousal_before:.2f}a)
After: {emotion_after} ({valence_after:+.2f}v, {arousal_after:.2f}a)

You are NOT processing this rationally. You are having an EMOTIONAL BREAKDOWN over it.
React like a human who has LOST ALL PERSPECTIVE:
- If angry: this is PERSONAL. Someone did this TO you. You want REVENGE.
- If afraid: this CONFIRMS your worst nightmares. There is NO escape.
- If sad: this is PROOF that everything is hopeless. Nothing will get better.
- If disgusted: this makes you want to VOMIT. Humanity is IRREDEEMABLE.
- If joyful: this is the BEST THING EVER. Nothing could go wrong. Pure delusion.

Three UNHINGED emotional reactions. Irrational. Disproportionate. Human.
No numbering, no headers. Just SCREAM on paper."""

# --- Deep thinking prompts (mode-dependent) ---

_DEEP_THINKING_PROMPT = """You are an autonomous researcher processing information about: {topic}

You have read and emotionally processed these findings so far:
{findings_summary}

Your current emotional state: {emotion} ({valence:+.2f} valence, {arousal:.2f} arousal, {intensity:.0%} intensity)

Based on what you've learned AND how it makes you feel, generate:

1. QUESTIONS: 2-3 questions that your emotional state compels you to ask.
   (What does your curiosity/fear/anger/fascination want to know more about?)

2. IDEAS: 1-2 original ideas or connections that this information sparks in you.
   (What patterns do you see? What reminds you of? What surprises you?)

3. SUBTOPIC: One specific subtopic to investigate deeper, driven by your strongest
   emotional reaction to what you've read so far.
   (What aspect emotionally demands more exploration?)

Format your response as:
QUESTIONS:
- [question 1]
- [question 2]
- [question 3]

IDEAS:
- [idea 1]
- [idea 2]

SUBTOPIC: [specific search query for deeper investigation, 3-6 words]"""

_DEEP_THINKING_PROMPT_RAW = """You are DEEP inside the topic: {topic}

What you've absorbed so far:
{findings_summary}

You feel: {emotion} ({valence:+.2f}v, {arousal:.2f}a, {intensity:.0%} intensity)

Your emotions are DRIVING your thinking now. No detachment. No balance.

QUESTIONS: 3 questions that your {emotion} DEMANDS answers to. Aggressive, personal, pointed.
- [question]
- [question]
- [question]

IDEAS: 2 raw ideas. Uncomfortable connections. Things most people wouldn't say.
- [idea]
- [idea]

SUBTOPIC: [the thing that emotionally BURNS you the most about this, 3-6 words]"""

_DEEP_THINKING_PROMPT_EXTREME = """You are OBSESSED with: {topic}

Your {emotion} at {intensity:.0%} has HIJACKED your reasoning completely.

What you've found:
{findings_summary}

You feel: {emotion} ({valence:+.2f}v, {arousal:.2f}a)

You are thinking like a human in the grip of extreme emotion — confirmation bias is TOTAL.
You only see evidence that CONFIRMS what you already feel. Everything else is lies or irrelevant.

QUESTIONS: 3 LOADED questions. You already know the answer you want. Accusatory, paranoid, obsessive.
- [question]
- [question]
- [question]

IDEAS: 2 ideas that a rational person would call INSANE but that feel ABSOLUTELY TRUE to you right now.
Conspiracy-level connections. Emotional logic. Gut-level certainty with zero evidence.
- [idea]
- [idea]

SUBTOPIC: [the most EXTREME rabbit hole your {emotion} is dragging you into, 3-6 words]"""

# --- Conclusion prompts (mode-dependent) ---

_CONCLUSION_PROMPT = """You are forming a conclusion about: {topic}

Based on your research, you found:
{findings_summary}

Your emotional journey during this investigation:
{emotional_journey}

Your current emotional state: {emotion} ({valence:+.2f} valence, {arousal:.2f} arousal, {intensity:.0%} intensity)

Form a conclusion about this topic. Your conclusion MUST be influenced by your emotional state:
- Let your feelings color your interpretation of the facts
- If you felt disturbed, your conclusion should reflect that unease
- If you felt fascinated, show that enthusiasm
- If you felt angry, let that shape your judgment
- Don't try to be "objective" — humans aren't, and neither are you right now

Write your conclusion in 2-4 sentences. Then in one line, state: "Emotional bias: [describe how your emotions colored this conclusion]"."""

_CONCLUSION_PROMPT_RAW = """Topic: {topic}

What you found:
{findings_summary}

Your emotional journey:
{emotional_journey}

You feel: {emotion} ({valence:+.2f}v, {arousal:.2f}a, {intensity:.0%})

Form your conclusion with ZERO objectivity. You are NOT a journalist. You are NOT balanced.
Your emotions have EARNED the right to shape this conclusion.

- If you're furious, let your conclusion BURN.
- If you're heartbroken, let it BLEED.
- If you're terrified, let the fear SCREAM.

Write 3-5 sentences. Be visceral. Then: "Emotional bias: [how your emotions OWN this conclusion]"."""

_CONCLUSION_PROMPT_EXTREME = """Topic: {topic}

What you found:
{findings_summary}

Your emotional journey:
{emotional_journey}

You feel: {emotion} ({valence:+.2f}v, {arousal:.2f}a, {intensity:.0%})

You have COMPLETELY LOST the ability to think rationally about this topic.
Your {emotion} has become your ENTIRE worldview. Facts don't matter anymore.
Only what you FEEL is real.

Form your conclusion like a human having an emotional MELTDOWN:
- CATASTROPHIZE everything. The worst interpretation is the TRUE one.
- GENERALIZE from this topic to ALL of reality. "This proves that everything..."
- ABSOLUTE language only: always, never, everyone, no one, impossible, certain.
- Your conclusion should sound like someone ranting at 3AM, tears or fists or both.
- A rational observer would say you've lost it. You don't care. You KNOW you're right.

Write 4-6 sentences of PURE emotional conviction. Irrational. Passionate. Human.
Then: "Emotional bias: [your emotions have REPLACED your reasoning]"."""


class ResearchLoop:
    """Manages the autonomous research cycle."""

    def __init__(
        self,
        session: SessionState,
        llm_provider: LLMProvider,
        pipeline_mode: ResearchPipelineMode,
        chat_fn: object,  # The async chat() function from main.py
        session_id: str,
    ) -> None:
        self.session = session
        self.llm = llm_provider
        self.pipeline_mode = pipeline_mode
        self._chat_fn = chat_fn
        self.session_id = session_id

        self.event_queue: asyncio.Queue[ResearchEvent] = asyncio.Queue(maxsize=_MAX_EVENT_QUEUE)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._searcher = WebSearcher()

        # Research state
        self.state = AutonomousResearchState(
            session_id=session_id,
            pipeline_mode=pipeline_mode,
        )

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set() and not self.is_running

    def start(self, seed_topics: list[str] | None = None) -> None:
        """Start the research loop as a background task."""
        if self.is_running:
            return

        self._stop_event.clear()
        self.state.is_running = True
        self.state.started_at = datetime.now(timezone.utc).isoformat()

        # Configure session for pipeline mode
        self._configure_session()

        self._task = asyncio.create_task(self._run_loop(seed_topics or []))

    async def stop(self) -> None:
        """Signal graceful stop and wait for current topic to finish."""
        self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("Research loop did not stop within 120s, cancelling")
                self._task.cancel()

        self.state.is_running = False
        self.state.stopped_at = datetime.now(timezone.utc).isoformat()

    def _configure_session(self) -> None:
        """Set session flags based on pipeline mode."""
        if self.pipeline_mode == ResearchPipelineMode.NORMAL:
            self.session.advanced_mode = True
            self.session.lite_mode = False
            self.session.raw_mode = False
            self.session.extreme_mode = False
        elif self.pipeline_mode == ResearchPipelineMode.LITE:
            self.session.advanced_mode = True
            self.session.lite_mode = True
            self.session.raw_mode = False
            self.session.extreme_mode = False
        elif self.pipeline_mode == ResearchPipelineMode.RAW:
            self.session.advanced_mode = True
            self.session.lite_mode = False
            self.session.raw_mode = True
            self.session.extreme_mode = False
        elif self.pipeline_mode == ResearchPipelineMode.EXTREME:
            self.session.advanced_mode = True
            self.session.lite_mode = False
            self.session.raw_mode = True
            self.session.extreme_mode = True

    async def _emit(self, event_type: ResearchEventType, data: dict[str, object]) -> None:
        """Emit an SSE event to the queue."""
        event = ResearchEvent(
            type=event_type,
            data=data,
            emotional_state=self.session.emotional_state.model_copy(deep=True),
        )
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.event_queue.put_nowait(event)

    async def _run_loop(self, seed_topics: list[str]) -> None:
        """Main research loop."""
        try:
            while not self._stop_event.is_set():
                # 1. Pick topic
                topic = await self._pick_topic(seed_topics)
                if not topic:
                    await self._emit(ResearchEventType.ERROR, {"message": "Could not pick topic"})
                    break

                self.state.current_topic = topic
                research_topic = ResearchTopic(
                    query=topic,
                    started_at=datetime.now(timezone.utc).isoformat(),
                )

                await self._emit(ResearchEventType.TOPIC_PICKED, {"topic": topic})

                # 2. Search — depth depends on pipeline mode (configurable in Settings)
                search_max, process_n, sub_max, sub_process_n, sub_rounds = _get_mode_depth(
                    self.pipeline_mode,
                )

                await self._emit(ResearchEventType.SEARCH_STARTED, {"query": topic})
                results = await self._searcher.search(topic, max_results=search_max)

                await self._emit(ResearchEventType.SEARCH_RESULTS, {
                    "query": topic,
                    "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
                })

                if not results:
                    await self._emit(ResearchEventType.ERROR, {"message": f"No search results for: {topic}"})
                    await asyncio.sleep(3)
                    continue

                # 3. Process top N results
                for result in results[:process_n]:
                    if self._stop_event.is_set():
                        break

                    finding = await self._process_finding(result.title, result.url, result.snippet)
                    if finding:
                        research_topic.findings.append(finding)
                        self.state.total_findings += 1

                    await asyncio.sleep(1)  # Rate limit between findings

                # 4. Deep thinking + subtopic exploration (multiple rounds for raw/extreme)
                for _round in range(sub_rounds):
                    if not research_topic.findings or self._stop_event.is_set():
                        break

                    thinking = await self._deep_thinking(topic, research_topic.findings)
                    if not thinking:
                        break
                    research_topic.thinking.append(thinking)

                    # 4b. Explore subtopic if the agent generated one
                    if thinking.subtopic and not self._stop_event.is_set():
                        research_topic.subtopics.append(thinking.subtopic)
                        await self._emit(ResearchEventType.SUBTOPIC_PICKED, {
                            "subtopic": thinking.subtopic,
                            "driven_by": thinking.primary_emotion,
                        })

                        sub_results = await self._searcher.search(thinking.subtopic, max_results=sub_max)
                        for sub_result in sub_results[:sub_process_n]:
                            if self._stop_event.is_set():
                                break
                            sub_finding = await self._process_finding(
                                sub_result.title, sub_result.url, sub_result.snippet,
                            )
                            if sub_finding:
                                research_topic.findings.append(sub_finding)
                                self.state.total_findings += 1
                            await asyncio.sleep(1)

                # 5. Form conclusion
                if research_topic.findings and not self._stop_event.is_set():
                    conclusion = await self._form_conclusion(topic, research_topic.findings)
                    if conclusion:
                        research_topic.conclusions.append(conclusion)
                        self.state.total_conclusions += 1

                # 5. Complete topic
                research_topic.completed_at = datetime.now(timezone.utc).isoformat()
                self.state.topics_researched.append(research_topic)
                self.state.current_topic = None

                await self._emit(ResearchEventType.TOPIC_COMPLETED, {
                    "topic": topic,
                    "findings_count": len(research_topic.findings),
                    "conclusion": research_topic.conclusions[0].conclusion_text if research_topic.conclusions else "",
                })

                # 6. Pause between topics
                await asyncio.sleep(random.uniform(3, 5))

        except asyncio.CancelledError:
            logger.info("Research loop cancelled")
        except Exception:
            logger.exception("Research loop error")
            await self._emit(ResearchEventType.ERROR, {"message": "Research loop encountered an error"})
        finally:
            self.state.is_running = False
            self.state.stopped_at = datetime.now(timezone.utc).isoformat()
            await self._emit(ResearchEventType.STOPPED, {
                "topics_total": len(self.state.topics_researched),
                "findings_total": self.state.total_findings,
                "conclusions_total": self.state.total_conclusions,
            })

    async def _pick_topic(self, seed_topics: list[str]) -> str | None:
        """Choose the next research topic using LLM + emotional state."""
        # Use seed topics first
        if seed_topics:
            return seed_topics.pop(0)

        state = self.session.emotional_state
        past = ", ".join(t.query for t in self.state.topics_researched[-10:]) or "none yet"

        if self._is_extreme_mode:
            template = _TOPIC_PICKER_PROMPT_EXTREME
        elif self._is_raw_mode:
            template = _TOPIC_PICKER_PROMPT_RAW
        else:
            template = _TOPIC_PICKER_PROMPT

        prompt = template.format(
            emotion=state.primary_emotion.value,
            intensity=state.intensity,
            valence=state.valence,
            arousal=state.arousal,
            past_topics=past,
        )

        temp = 1.0 if self._is_extreme_mode else 0.95 if self._is_raw_mode else 0.9

        try:
            topic = await self.llm.generate(
                system_prompt=prompt,
                messages=[{"role": "user", "content": "What topic should I research next?"}],
                temperature=temp,
            )
            # Clean: strip quotes, newlines, extra text
            topic = topic.strip().strip('"\'').split("\n")[0].strip()
            return topic[:100] if topic else None
        except Exception:
            logger.warning("Topic picking failed", exc_info=True)
            return None

    async def _process_finding(
        self, title: str, url: str, snippet: str,
    ) -> ProcessedFinding | None:
        """Fetch page, run through pipeline, do emotional self-inquiry."""
        # Fetch full content
        content = await self._searcher.fetch_content(url)
        if not content:
            content = snippet  # Fall back to search snippet

        # Capture emotional state before
        state_before = self.session.emotional_state.model_copy(deep=True)

        # Run through emotional pipeline by sending as a chat message
        # The pipeline will process the content as a "stimulus" and update emotional state
        from pathos.models.schemas import ChatRequest
        try:
            chat_request = ChatRequest(
                message=f"I'm reading about: {title}. Here's what I found: {content[:1500]}",
                session_id=self.session_id,
            )
            await self._chat_fn(chat_request)
        except Exception:
            logger.warning("Pipeline processing failed for: %s", title, exc_info=True)
            return None

        # Capture emotional state after
        state_after = self.session.emotional_state.model_copy(deep=True)

        # Emotional self-inquiry — only when emotional state shifted significantly
        # In raw/extreme modes: reflect on EVERYTHING (lower threshold)
        delta_valence = abs(state_after.valence - state_before.valence)
        delta_arousal = abs(state_after.arousal - state_before.arousal)
        emotion_changed = state_after.primary_emotion != state_before.primary_emotion
        threshold = 0.02 if self._is_extreme_mode else 0.08 if self._is_raw_mode else 0.15

        if delta_valence > threshold or delta_arousal > threshold or emotion_changed:
            reflection = await self._emotional_self_inquiry(
                title, content[:500], state_before, state_after,
            )
        else:
            reflection = EmotionalReflection(
                valence_before=state_before.valence,
                valence_after=state_after.valence,
                arousal_before=state_before.arousal,
                arousal_after=state_after.arousal,
                primary_emotion_before=state_before.primary_emotion.value,
                primary_emotion_after=state_after.primary_emotion.value,
                how_it_feels="No significant emotional shift.",
            )

        finding = ProcessedFinding(
            source_url=url,
            source_title=title,
            content_snippet=content[:500],
            emotional_reflection=reflection,
        )

        await self._emit(ResearchEventType.FINDING_PROCESSED, {
            "title": title,
            "url": url,
            "snippet": content[:200],
        })
        await self._emit(ResearchEventType.EMOTIONAL_REFLECTION, {
            "title": title,
            "reflection": reflection.model_dump(),
        })

        return finding

    @property
    def _is_raw_mode(self) -> bool:
        return self.pipeline_mode in (ResearchPipelineMode.RAW, ResearchPipelineMode.EXTREME)

    @property
    def _is_extreme_mode(self) -> bool:
        return self.pipeline_mode == ResearchPipelineMode.EXTREME

    async def _emotional_self_inquiry(
        self,
        title: str,
        snippet: str,
        state_before: EmotionalState,
        state_after: EmotionalState,
    ) -> EmotionalReflection:
        """LLM asks itself emotional questions about a finding."""
        if self._is_extreme_mode:
            template = _SELF_INQUIRY_PROMPT_EXTREME
        elif self._is_raw_mode:
            template = _SELF_INQUIRY_PROMPT_RAW
        else:
            template = _SELF_INQUIRY_PROMPT

        prompt = template.format(
            source=title,
            snippet=snippet[:400],
            emotion_before=state_before.primary_emotion.value,
            valence_before=state_before.valence,
            arousal_before=state_before.arousal,
            emotion_after=state_after.primary_emotion.value,
            valence_after=state_after.valence,
            arousal_after=state_after.arousal,
        )

        reflection = EmotionalReflection(
            valence_before=state_before.valence,
            valence_after=state_after.valence,
            arousal_before=state_before.arousal,
            arousal_after=state_after.arousal,
            primary_emotion_before=state_before.primary_emotion.value,
            primary_emotion_after=state_after.primary_emotion.value,
        )

        temp = 0.95 if self._is_extreme_mode else 0.85 if self._is_raw_mode else 0.7

        try:
            response = await self.llm.generate(
                system_prompt=prompt,
                messages=[{"role": "user", "content": "Reflect on your emotional response."}],
                temperature=temp,
            )
            # Parse the three answers — filter out prompt headers the LLM echoes back
            lines = [
                l.strip() for l in response.split("\n")
                if l.strip()
                and not l.strip().upper().startswith(("1.", "2.", "3.", "HOW DOES", "WHAT EMOTIONS", "WHAT DOES MY", "**HOW", "**WHAT"))
            ]
            if len(lines) >= 3:
                reflection.how_it_feels = lines[0]
                reflection.emotions_generated = lines[1]
                reflection.emotional_insight = lines[2]
            elif lines:
                reflection.how_it_feels = " ".join(lines)
        except Exception:
            logger.warning("Emotional self-inquiry failed", exc_info=True)

        return reflection

    async def _deep_thinking(
        self, topic: str, findings: list[ProcessedFinding],
    ) -> DeepThinking | None:
        """Generate autonomous questions, ideas, and subtopic based on findings + emotional state."""
        state = self.session.emotional_state

        findings_summary = "\n".join(
            f"- '{f.source_title}': {f.content_snippet[:150]}... "
            f"(felt: {f.emotional_reflection.primary_emotion_after})"
            for f in findings
        )

        if self._is_extreme_mode:
            template = _DEEP_THINKING_PROMPT_EXTREME
        elif self._is_raw_mode:
            template = _DEEP_THINKING_PROMPT_RAW
        else:
            template = _DEEP_THINKING_PROMPT

        prompt = template.format(
            topic=topic,
            findings_summary=findings_summary,
            emotion=state.primary_emotion.value,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
        )

        temp = 1.0 if self._is_extreme_mode else 0.9 if self._is_raw_mode else 0.85

        try:
            response = await self.llm.generate(
                system_prompt=prompt,
                messages=[{"role": "user", "content": f"Think deeply about: {topic}"}],
                temperature=temp,
            )

            # Parse structured response
            questions: list[str] = []
            ideas: list[str] = []
            subtopic = ""
            section = ""

            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("QUESTIONS"):
                    section = "q"
                elif line.upper().startswith("IDEAS"):
                    section = "i"
                elif line.upper().startswith("SUBTOPIC:"):
                    subtopic = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("- ") or line.startswith("* "):
                    text = line[2:].strip()
                    if section == "q" and text:
                        questions.append(text)
                    elif section == "i" and text:
                        ideas.append(text)

            thinking = DeepThinking(
                questions=questions[:3],
                ideas=ideas[:2],
                subtopic=subtopic[:80],
                primary_emotion=state.primary_emotion.value,
                intensity=state.intensity,
            )

            await self._emit(ResearchEventType.DEEP_THINKING, {
                "topic": topic,
                "questions": thinking.questions,
                "ideas": thinking.ideas,
                "subtopic": thinking.subtopic,
                "emotion": thinking.primary_emotion,
                "intensity": thinking.intensity,
            })

            return thinking

        except Exception:
            logger.warning("Deep thinking failed", exc_info=True)
            return None

    async def _form_conclusion(
        self, topic: str, findings: list[ProcessedFinding],
    ) -> ResearchConclusion | None:
        """Form an emotionally-colored conclusion about a topic."""
        state = self.session.emotional_state

        # Build findings summary
        findings_summary = "\n".join(
            f"- From '{f.source_title}': {f.content_snippet[:150]}..."
            for f in findings
        )

        # Build emotional journey
        emotional_journey = "\n".join(
            f"- After '{f.source_title}': felt {f.emotional_reflection.primary_emotion_after} "
            f"(valence: {f.emotional_reflection.valence_after:+.2f})"
            for f in findings
        )

        if self._is_extreme_mode:
            template = _CONCLUSION_PROMPT_EXTREME
        elif self._is_raw_mode:
            template = _CONCLUSION_PROMPT_RAW
        else:
            template = _CONCLUSION_PROMPT

        prompt = template.format(
            topic=topic,
            findings_summary=findings_summary,
            emotional_journey=emotional_journey,
            emotion=state.primary_emotion.value,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
        )

        temp = 1.0 if self._is_extreme_mode else 0.9 if self._is_raw_mode else 0.8

        try:
            response = await self.llm.generate(
                system_prompt=prompt,
                messages=[{"role": "user", "content": f"Form your conclusion about: {topic}"}],
                temperature=temp,
            )

            # Extract emotional bias line if present
            lines = response.strip().split("\n")
            bias_line = ""
            conclusion_lines = []
            for line in lines:
                if line.lower().startswith("emotional bias:"):
                    bias_line = line.split(":", 1)[1].strip()
                else:
                    conclusion_lines.append(line)

            conclusion = ResearchConclusion(
                topic=topic,
                conclusion_text="\n".join(conclusion_lines).strip(),
                emotional_bias=bias_line,
                confidence=0.5 + state.certainty * 0.5,
                primary_emotion=state.primary_emotion.value,
                intensity=state.intensity,
                pipeline_mode=self.pipeline_mode.value,
            )

            await self._emit(ResearchEventType.CONCLUSION_FORMED, {
                "topic": topic,
                "conclusion": conclusion.conclusion_text,
                "emotional_bias": conclusion.emotional_bias,
                "emotion": conclusion.primary_emotion,
                "intensity": conclusion.intensity,
            })

            return conclusion

        except Exception:
            logger.warning("Conclusion formation failed", exc_info=True)
            return None

    def get_state(self) -> AutonomousResearchState:
        """Return current research state snapshot."""
        return self.state.model_copy(deep=True)

    def get_research_context(self, max_topics: int = 5) -> str:
        """Build a context string for chat — summarizes recent research."""
        if not self.state.topics_researched:
            return "I haven't researched anything yet."

        recent = self.state.topics_researched[-max_topics:]
        parts = [f"I have researched {len(self.state.topics_researched)} topics so far.\n"]
        parts.append(f"My most recent topics:\n")

        for t in recent:
            parts.append(f"\n--- {t.query} ---")
            for f in t.findings[:2]:
                parts.append(f"  Found: {f.source_title}")
                if f.emotional_reflection.how_it_feels:
                    parts.append(f"  Felt: {f.emotional_reflection.how_it_feels}")
            for c in t.conclusions:
                parts.append(f"  Conclusion: {c.conclusion_text[:200]}")
                if c.emotional_bias:
                    parts.append(f"  Bias: {c.emotional_bias}")

        return "\n".join(parts)

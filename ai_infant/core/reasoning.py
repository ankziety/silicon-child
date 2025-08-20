"""Reasoning engine that shows AI thinking process and decision-making."""

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

from ..learn.llm_aggregator import AggregatorManager
from ..text.ai_response import AIResponseGenerator


@dataclass
class Thought:
    """A single thought in the reasoning process."""

    id: str
    timestamp: datetime
    thought_type: (
        str  # "observation", "hypothesis", "question", "conclusion", "decision"
    )
    content: str
    confidence: float  # 0.0 to 1.0
    evidence: list[str]  # URLs or document IDs that support this thought
    parent_thought: Optional[str] = None  # ID of parent thought
    children: list[str] = None  # IDs of child thoughts

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class KnowledgeGap:
    """A gap in knowledge that needs to be filled."""

    id: str
    question: str
    importance: float  # 0.0 to 1.0
    related_thoughts: list[str]  # Thought IDs that led to this gap
    search_queries: list[str]  # Suggested search queries
    created_at: datetime
    filled: bool = False
    filled_at: Optional[datetime] = None


@dataclass
class WorkingMemory:
    """The AI's working memory that gets updated during research."""

    id: str
    created_at: datetime
    updated_at: datetime
    facts: dict[str, Any]  # Key-value pairs of learned facts
    hypotheses: list[str]  # Current hypotheses being tested
    conclusions: list[str]  # Formed conclusions
    confidence_scores: dict[str, float]  # Confidence in each fact/hypothesis
    contradictions: list[tuple[str, str]]  # Conflicting information
    sources: dict[str, list[str]]  # Sources for each fact


class ReasoningEngine:
    """Engine that manages AI reasoning and shows thinking process."""

    def __init__(
        self,
        store: Any,
        ai_generator: AIResponseGenerator,
        aggregator: Optional[AggregatorManager] = None,
    ):
        self.store = store
        self.ai_generator = ai_generator
        self.aggregator = aggregator
        self.thoughts: list[Thought] = []
        self.knowledge_gaps: list[KnowledgeGap] = []
        self.working_memory = WorkingMemory(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            facts={},
            hypotheses=[],
            conclusions=[],
            confidence_scores={},
            contradictions=[],
            sources={},
        )

    def log_thought(
        self,
        thought_type: str,
        content: str,
        confidence: float = 0.5,
        evidence: list[str] = None,
        parent_thought: Optional[str] = None,
    ) -> str:
        """Log a new thought and show it in the reasoning process."""
        thought_id = str(uuid.uuid4())

        thought = Thought(
            id=thought_id,
            timestamp=datetime.utcnow(),
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            evidence=evidence or [],
            parent_thought=parent_thought,
        )

        self.thoughts.append(thought)

        # Add to parent's children if applicable
        if parent_thought:
            for parent in self.thoughts:
                if parent.id == parent_thought:
                    parent.children.append(thought_id)
                    break

        # Log the thought
        self._log_reasoning_step(thought)

        return thought_id

    def _log_reasoning_step(self, thought: Thought):
        """Log a reasoning step with detailed information."""
        print(f"\nREASONING: {thought.thought_type.upper()}")
        print(f"   Content: {thought.content}")
        print(f"   Confidence: {thought.confidence:.2f}")
        if thought.evidence:
            print(f"   Evidence: {len(thought.evidence)} sources")
        if thought.parent_thought:
            print(f"   Based on: {thought.parent_thought}")
        print()

        # Store in database
        self.store.store_trace({
            "id": f"reasoning-{thought.id}",
            "job_id": f"reasoning-{int(time.time() * 1000)}",
            "component": "reasoning",
            "operation": thought.thought_type,
            "status": "completed",
            "timestamp": thought.timestamp.isoformat() + "Z",
            "duration_ms": 0,
            "input": {"thought_type": thought.thought_type, "content": thought.content},
            "output": {"thought_id": thought.id, "confidence": thought.confidence},
            "metadata": {"reasoning_step": True},
        })

    def analyze_content(self, content: str, source_url: str) -> str:
        """Analyze new content and form thoughts about it."""
        thought_id = self.log_thought(
            "observation",
            f"Analyzing content from {source_url}",
            confidence=0.8,
            evidence=[source_url],
        )

        # Extract key information using actual text analysis
        extracted_info = self._extract_key_information(content)

        # Process extracted facts
        for fact in extracted_info["facts"]:
            fact_id = f"fact_{len(self.working_memory.facts)}"
            self.working_memory.facts[fact_id] = fact["text"]
            self.working_memory.confidence_scores[fact_id] = fact["confidence"]
            self.working_memory.sources[fact_id] = [source_url]

            self.log_thought(
                "observation",
                f"Learned: {fact['text']}",
                confidence=fact["confidence"],
                evidence=[source_url],
                parent_thought=thought_id,
            )

        # Generate hypotheses based on content
        hypotheses = self._generate_hypotheses(content, extracted_info["facts"])
        for hypothesis in hypotheses:
            self.working_memory.hypotheses.append(hypothesis)
            self.log_thought(
                "hypothesis",
                f"Hypothesis: {hypothesis}",
                confidence=0.6,
                evidence=[source_url],
                parent_thought=thought_id,
            )

        # Identify knowledge gaps based on content analysis
        gaps = self._identify_content_gaps(content, extracted_info["facts"])
        for gap_question in gaps:
            self.identify_knowledge_gap(gap_question, [thought_id])
            self.log_thought(
                "question",
                f"Need to know: {gap_question}",
                confidence=0.9,
                evidence=[source_url],
                parent_thought=thought_id,
            )

        # Update working memory timestamp
        self.working_memory.updated_at = datetime.utcnow()

        return thought_id

    def _extract_key_information(self, content: str) -> dict:
        """Extract key information from content using AI analysis."""
        prompt = f"""<analysis_request>
<content>
{content}
</content>
<task>Extract key facts and insights from this content. Be thorough and analytical.</task>
<output_format>
<facts>
  <fact confidence="0.0-1.0">fact text</fact>
</facts>
<important_terms>
  <term>term1</term>
  <term>term2</term>
</important_terms>
<relevant_sentences>
  <sentence>sentence1</sentence>
  <sentence>sentence2</sentence>
</relevant_sentences>
</output_format>
</analysis_request>"""

        try:
            response = self._route_llm(prompt, use_context=False)
            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response)

            facts = []
            for fact_elem in root.findall(".//fact"):
                facts.append(
                    {
                        "text": fact_elem.text,
                        "confidence": float(fact_elem.get("confidence", 0.7)),
                    }
                )

            important_terms = [term.text for term in root.findall(".//term")]
            relevant_sentences = [sent.text for sent in root.findall(".//sentence")]

            return {
                "facts": facts,
                "important_terms": important_terms,
                "relevant_sentences": relevant_sentences,
            }
        except Exception:
            # Fallback to basic extraction if AI fails
            return self._fallback_extract_key_information(content)

    def _fallback_extract_key_information(self, content: str) -> dict:
        """Fallback key information extraction using basic text analysis."""
        import re
        from collections import Counter

        content_lower = content.lower()
        words = re.findall(r"\b\w+\b", content_lower)

        key_terms = []
        for word in words:
            if len(word) > 4 and word not in [
                "about",
                "their",
                "there",
                "these",
                "those",
                "which",
                "where",
                "would",
                "could",
                "should",
            ]:
                key_terms.append(word)

        term_freq = Counter(key_terms)
        important_terms = [term for term, freq in term_freq.most_common(10) if freq > 2]

        sentences = re.split(r"[.!?]+", content)
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if (
                any(term in sentence.lower() for term in important_terms)
                and len(sentence) > 20
            ):
                relevant_sentences.append(sentence)

        facts = []
        for sentence in relevant_sentences[:5]:
            if any(
                word in sentence.lower()
                for word in [
                    "is",
                    "are",
                    "was",
                    "were",
                    "has",
                    "have",
                    "can",
                    "will",
                    "does",
                ]
            ):
                facts.append(
                    {
                        "text": sentence.strip(),
                        "confidence": 0.7 if len(sentence) > 50 else 0.5,
                    }
                )

        return {
            "facts": facts,
            "important_terms": important_terms,
            "relevant_sentences": relevant_sentences,
        }

    def _generate_hypotheses(self, content: str, facts: list) -> list:
        """Generate hypotheses based on content and facts using AI analysis."""
        fact_texts = [fact["text"] for fact in facts]

        prompt = f"""<hypothesis_generation>
<content>
{content}
</content>
<extracted_facts>
{chr(10).join(fact_texts)}
</extracted_facts>
<task>Generate 3-5 testable hypotheses based on the content and facts. Think critically about what connections, patterns, or implications might exist.</task>
<output_format>
<hypotheses>
  <hypothesis confidence="0.0-1.0">hypothesis text</hypothesis>
</hypotheses>
</output_format>
</hypothesis_generation>"""

        try:
            response = self._route_llm(prompt, use_context=False)
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response)

            hypotheses = []
            for hyp_elem in root.findall(".//hypothesis"):
                hypotheses.append(hyp_elem.text)

            return hypotheses
        except Exception:
            # Fallback to basic hypothesis generation
            return self._fallback_generate_hypotheses(content, facts)

    # HATE: I hate that this has to exist, very 
    def _fallback_generate_hypotheses(self, content: str, facts: list) -> list:
        """Fallback hypothesis generation using basic pattern matching."""
        hypotheses = []
        content_lower = content.lower()

        if len(facts) > 2:
            fact_texts = [fact["text"] for fact in facts]
            common_terms = set()
            for fact in fact_texts:
                words = fact.lower().split()
                common_terms.update([w for w in words if len(w) > 4])

            if len(common_terms) > 2:
                hypotheses.append(
                    f"There may be connections between {', '.join(list(common_terms)[:3])}"
                )

        problem_words = [
            "problem",
            "issue",
            "challenge",
            "difficulty",
            "obstacle",
            "limitation",
        ]
        if any(word in content_lower for word in problem_words):
            hypotheses.append(
                "There may be solutions or approaches to address the mentioned challenges"
            )

        trend_words = [
            "trend",
            "growing",
            "increasing",
            "emerging",
            "developing",
            "future",
        ]
        if any(word in content_lower for word in trend_words):
            hypotheses.append("This trend may continue or accelerate in the future")

        return hypotheses

    def _identify_content_gaps(self, content: str, facts: list) -> list:
        """Identify knowledge gaps based on content analysis using AI."""
        fact_texts = [fact["text"] for fact in facts]

        prompt = f"""<gap_analysis>
<content>
{content}
</content>
<extracted_facts>
{chr(10).join(fact_texts)}
</extracted_facts>
<task>Identify 3-5 specific knowledge gaps or unanswered questions from this content. Look for missing explanations, incomplete information, or areas that need further research.</task>
<output_format>
<knowledge_gaps>
  <gap importance="0.0-1.0">specific question about missing information</gap>
</knowledge_gaps>
</output_format>
</gap_analysis>"""

        try:
            # gaps may benefit from contextual generation; mark use_context True
            response = self._route_llm(prompt, use_context=True)
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response)

            gaps = []
            for gap_elem in root.findall(".//gap"):
                gaps.append(gap_elem.text)

            return gaps
        except Exception:
            # Fallback to basic gap identification
            return self._fallback_identify_content_gaps(content, facts)

    def _fallback_identify_content_gaps(self, content: str, facts: list) -> list:
        """Fallback knowledge gap identification using basic pattern matching."""
        gaps = []
        content_lower = content.lower()

        technical_terms = [
            "algorithm",
            "protocol",
            "framework",
            "methodology",
            "paradigm",
        ]
        mentioned_terms = [term for term in technical_terms if term in content_lower]

        for term in mentioned_terms:
            if (
                f"what is {term}" not in content_lower
                and f"how {term}" not in content_lower
            ):
                gaps.append(f"What is {term} and how does it work?")

        if any(word in content_lower for word in ["problem", "issue", "challenge"]):
            if "solution" not in content_lower and "approach" not in content_lower:
                gaps.append(
                    "What are the solutions or approaches to address these challenges?"
                )

        if any(
            word in content_lower for word in ["current", "now", "today", "present"]
        ):
            if "future" not in content_lower and "next" not in content_lower:
                gaps.append("What are the future developments or next steps?")

        if any(
            word in content_lower
            for word in ["benefit", "advantage", "improve", "better"]
        ):
            if "drawback" not in content_lower and "disadvantage" not in content_lower:
                gaps.append("What are the potential drawbacks or limitations?")

        return gaps

    def identify_knowledge_gap(self, question: str, related_thoughts: list[str]) -> str:
        """Identify a knowledge gap and create a search target."""
        gap_id = str(uuid.uuid4())

        # Generate search queries based on the question content
        search_queries = self._generate_search_queries(question)

        # Calculate importance based on question content and context
        importance = self._calculate_gap_importance(question, related_thoughts)

        gap = KnowledgeGap(
            id=gap_id,
            question=question,
            importance=importance,
            related_thoughts=related_thoughts,
            search_queries=search_queries,
            created_at=datetime.utcnow(),
        )

        self.knowledge_gaps.append(gap)

        print("\n❓ KNOWLEDGE GAP IDENTIFIED:")
        print(f"   Question: {question}")
        print(f"   Search queries: {search_queries}")
        print(f"   Importance: {importance:.2f}")
        print()

        return gap_id

    def _generate_search_queries(self, question: str) -> list[str]:
        """Generate search queries based on question content."""
        import re

        # Extract key terms from the question
        question_lower = question.lower()
        words = re.findall(r"\b\w+\b", question_lower)

        # Filter out common words and short words
        key_terms = [
            word
            for word in words
            if len(word) > 3
            and word
            not in [
                "what",
                "when",
                "where",
                "which",
                "that",
                "this",
                "with",
                "from",
                "they",
                "have",
                "been",
                "will",
                "were",
                "about",
                "their",
                "there",
            ]
        ]

        # Generate different types of search queries
        queries = []

        # Query 1: Direct question
        queries.append(question)

        # Query 2: Key terms focused
        if key_terms:
            queries.append(f"{' '.join(key_terms[:3])} latest developments")

        # Query 3: Research focused
        if "what" in question_lower:
            # Replace "what" with more specific terms
            if "how" in question_lower:
                queries.append(
                    f"how to {question_lower.replace('what', '').replace('how', '').strip()}"
                )
            else:
                queries.append(f"research {question_lower.replace('what', '').strip()}")

        # Query 4: Technical focus
        technical_words = [
            "technology",
            "algorithm",
            "system",
            "method",
            "approach",
            "technique",
        ]
        if any(word in question_lower for word in technical_words):
            queries.append(f"technical details {question_lower}")

        # Ensure we have at least 3 queries
        while len(queries) < 3:
            queries.append(f"information about {question}")

        return queries[:3]

    def _route_llm(self, prompt: str, use_context: bool = False) -> str:
        """Route LLM call through aggregator if available, otherwise use AIResponseGenerator.

        use_context indicates whether to use contextual generation (generate_response) vs direct.
        """
        try:
            if self.aggregator:
                return self.aggregator.generate(prompt)
            else:
                if use_context:
                    return self.ai_generator.generate_response(prompt)
                else:
                    # direct generation without aggregator
                    return self.ai_generator.generate_direct_response(prompt)
        except Exception:
            # Re-raise so callers can fallback if desired
            raise

    def _calculate_gap_importance(
        self, question: str, related_thoughts: list[str]
    ) -> float:
        """Calculate the importance of a knowledge gap."""
        importance = 0.5  # Base importance

        question_lower = question.lower()

        # Increase importance for technical questions
        technical_terms = ["how", "why", "method", "algorithm", "technique", "approach"]
        if any(term in question_lower for term in technical_terms):
            importance += 0.2

        # Increase importance for questions about problems/solutions
        problem_terms = ["problem", "issue", "challenge", "solution", "fix", "improve"]
        if any(term in question_lower for term in problem_terms):
            importance += 0.2

        # Increase importance for questions about current/future
        time_terms = ["current", "latest", "future", "trend", "developing", "emerging"]
        if any(term in question_lower for term in time_terms):
            importance += 0.1

        # Increase importance based on number of related thoughts
        if len(related_thoughts) > 2:
            importance += 0.1

        return min(importance, 1.0)  # Cap at 1.0

    def form_conclusion(self, hypothesis: str, evidence: list[str]) -> str:
        """Form a conclusion based on evidence and update working memory."""
        conclusion_id = str(uuid.uuid4())

        # Analyze hypothesis and evidence to form conclusion
        conclusion_data = self._analyze_hypothesis_evidence(hypothesis, evidence)

        conclusion = conclusion_data["conclusion"]
        confidence = conclusion_data["confidence"]
        reasoning = conclusion_data["reasoning"]

        # Add to working memory
        self.working_memory.conclusions.append(conclusion)
        self.working_memory.confidence_scores[conclusion_id] = confidence
        self.working_memory.sources[conclusion_id] = evidence

        # Log the conclusion
        thought_id = self.log_thought(
            "conclusion",
            f"Conclusion: {conclusion}\nReasoning: {reasoning}",
            confidence=confidence,
            evidence=evidence,
        )

        # Check if this fills any knowledge gaps
        self._check_gap_filling(conclusion, evidence)

        return conclusion_id

    def _analyze_hypothesis_evidence(
        self, hypothesis: str, evidence: list[str]
    ) -> dict:
        """Analyze hypothesis against evidence to form conclusion using AI."""
        evidence_text = "\n".join(evidence[:5])  # Use first 5 evidence sources

        prompt = f"""<conclusion_analysis>
<hypothesis>
{hypothesis}
</hypothesis>
<evidence_sources>
{evidence_text}
</evidence_sources>
<task>Analyze whether the evidence supports, contradicts, or provides mixed results for the hypothesis. Provide a detailed conclusion with reasoning.</task>
<output_format>
<conclusion>
  <verdict>supports/contradicts/mixed</verdict>
  <confidence>0.0-1.0</confidence>
  <conclusion_text>detailed conclusion about the hypothesis</conclusion_text>
  <reasoning>detailed reasoning for the conclusion</reasoning>
</conclusion>
</output_format>
</conclusion_analysis>"""

        try:
            response = self._route_llm(prompt, use_context=True)
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response)

            conclusion_elem = root.find(".//conclusion")
            if conclusion_elem is not None:
                verdict = conclusion_elem.find("verdict").text
                confidence = float(conclusion_elem.find("confidence").text)
                conclusion_text = conclusion_elem.find("conclusion_text").text
                reasoning = conclusion_elem.find("reasoning").text

                return {
                    "conclusion": conclusion_text,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
        except Exception:
            # Fallback to basic analysis
            return self._fallback_analyze_hypothesis_evidence(hypothesis, evidence)

    def _fallback_analyze_hypothesis_evidence(
        self, hypothesis: str, evidence: list[str]
    ) -> dict:
        """Fallback hypothesis analysis using basic pattern matching."""
        import re

        hypothesis_lower = hypothesis.lower()
        evidence_count = len(evidence)
        hypothesis_words = set(re.findall(r"\b\w+\b", hypothesis_lower))

        supporting_evidence = 0
        contradicting_evidence = 0

        for source in evidence:
            if any(word in source.lower() for word in hypothesis_words):
                supporting_evidence += 1
            else:
                contradicting_evidence += 1

        if evidence_count == 0:
            confidence = 0.1
            conclusion = f"Insufficient evidence to evaluate: {hypothesis}"
            reasoning = (
                "No evidence sources provided to support or contradict the hypothesis"
            )
        elif supporting_evidence > contradicting_evidence:
            confidence = min(0.3 + (supporting_evidence * 0.2), 0.9)
            conclusion = f"Evidence supports: {hypothesis}"
            reasoning = f"Found {supporting_evidence} supporting evidence sources vs {contradicting_evidence} contradicting"
        elif contradicting_evidence > supporting_evidence:
            confidence = 0.3
            conclusion = f"Evidence contradicts: {hypothesis}"
            reasoning = f"Found {contradicting_evidence} contradicting evidence sources vs {supporting_evidence} supporting"
        else:
            confidence = 0.5
            conclusion = f"Mixed evidence for: {hypothesis}"
            reasoning = f"Equal supporting ({supporting_evidence}) and contradicting ({contradicting_evidence}) evidence"

        return {
            "conclusion": conclusion,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def _check_gap_filling(self, conclusion: str, evidence: list[str]):
        """Check if a conclusion fills any knowledge gaps."""
        for gap in self.knowledge_gaps:
            if not gap.filled:
                # Simple check if conclusion relates to gap question
                if any(
                    word in conclusion.lower() for word in gap.question.lower().split()
                ):
                    gap.filled = True
                    gap.filled_at = datetime.utcnow()

                    self.log_thought(
                        "observation",
                        f"Knowledge gap filled: {gap.question}",
                        confidence=0.9,
                        evidence=evidence,
                    )

    def get_next_search_targets(self, max_targets: int = 3) -> list[KnowledgeGap]:
        """Get the most important unfilled knowledge gaps for searching."""
        unfilled_gaps = [gap for gap in self.knowledge_gaps if not gap.filled]

        # Sort by importance and age
        sorted_gaps = sorted(
            unfilled_gaps,
            key=lambda gap: (gap.importance, gap.created_at.timestamp()),
            reverse=True,
        )

        return sorted_gaps[:max_targets]

    def get_reasoning_summary(self) -> dict[str, Any]:
        """Get a summary of the reasoning process."""
        return {
            "total_thoughts": len(self.thoughts),
            "thought_types": {
                thought_type: len([t for t in self.thoughts if t.thought_type == thought_type])
                for thought_type in {t.thought_type for t in self.thoughts}
            },
            "knowledge_gaps": {
                "total": len(self.knowledge_gaps),
                "filled": len([g for g in self.knowledge_gaps if g.filled]),
                "unfilled": len([g for g in self.knowledge_gaps if not g.filled]),
            },
            "working_memory": {
                "facts_count": len(self.working_memory.facts),
                "hypotheses_count": len(self.working_memory.hypotheses),
                "conclusions_count": len(self.working_memory.conclusions),
                "contradictions_count": len(self.working_memory.contradictions),
            },
            "confidence_distribution": {
                "high": len([t for t in self.thoughts if t.confidence > 0.8]),
                "medium": len([t for t in self.thoughts if 0.5 <= t.confidence <= 0.8]),
                "low": len([t for t in self.thoughts if t.confidence < 0.5]),
            },
        }

    def export_reasoning_trace(self) -> dict[str, Any]:
        """Export the complete reasoning trace for analysis."""
        return {
            "reasoning_session": {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "working_memory": asdict(self.working_memory),
                "thoughts": [asdict(thought) for thought in self.thoughts],
                "knowledge_gaps": [asdict(gap) for gap in self.knowledge_gaps],
                "summary": self.get_reasoning_summary(),
            }
        }

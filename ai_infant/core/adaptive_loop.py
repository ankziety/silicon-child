"""Adaptive research loop with reasoning, learning, and iterative research."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import os
from typing import Any, Optional

from pydantic import BaseModel

from ..crawl.vision_browser import VisionBrowser, VisionModelConfig
from ..learn.continuous import ContinuousLearner
from ..text.ai_response import AIResponseGenerator
from ..text.image_analysis import ImageAnalyzer
from ..text.parse import Parser
from .reasoning import ReasoningEngine


class ResearchSession(BaseModel):
    """A complete research session with reasoning and learning."""

    id: str
    question: str
    created_at: datetime
    updated_at: datetime
    status: str  # "active", "completed", "failed"
    reasoning_summary: dict[str, Any]
    learning_stats: dict[str, Any]
    conclusions: list[str]
    sources_used: list[str]
    total_iterations: int
    final_answer: Optional[str] = None


class AdaptiveResearchLoop:
    """Research loop that thinks, learns, and researches iteratively."""

    def __init__(self, store: Any, headless: bool = False):
        """Initialize adaptive research loop."""
        self.store = store

        # Initialize vision browser with vision model configuration
        vision_config = VisionModelConfig(
            model_provider="openai",
            model_name="gpt-4o-mini",
            api_key=None,  # Will be loaded from environment
            api_base=None,  # Will use default API base
        )
        # Initialize system-wide aggregator and inject into browser
        from ..learn.llm_aggregator import AggregatorManager

        pref = os.getenv("AGGREGATOR_PREFERENCE", "llmz,openrouter")
        self.aggregator = AggregatorManager(preference=pref)
        self.browser = VisionBrowser(
            store, vision_config=vision_config, headless=headless
        )
        self.browser.aggregator = self.aggregator

        self.parser = Parser(store)
        self.ai_generator = AIResponseGenerator(store)
        self.image_analyzer = ImageAnalyzer(store)

        # New components
        self.reasoning_engine = ReasoningEngine(store, self.ai_generator)
        self.continuous_learner = ContinuousLearner(store)

        # Session state
        self.current_session: Optional[ResearchSession] = None
        self.iteration_count = 0
        self.max_iterations = 20

    def research_question(self, question: str) -> ResearchSession:
        """Research a question with full reasoning and learning."""
        print(f"\n🔍 STARTING RESEARCH: {question}")
        print("=" * 80)

        # Initialize research session
        self.current_session = ResearchSession(
            id=str(uuid.uuid4()),
            question=question,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status="active",
            reasoning_summary={},
            learning_stats={},
            conclusions=[],
            sources_used=[],
            total_iterations=0,
        )

        # Start reasoning about the question
        self.reasoning_engine.log_thought(
            "observation", f"Starting research on: {question}", confidence=1.0
        )

        # Generate initial search queries
        self._generate_initial_queries(question)

        # Main research loop
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n🔄 ITERATION {self.iteration_count}/{self.max_iterations}")
            print("-" * 60)

            # Get next search targets from reasoning engine
            search_targets = self.reasoning_engine.get_next_search_targets(
                max_targets=3
            )

            if not search_targets:
                print("No more search targets - research complete")
                break

            # Research each target
            for target in search_targets:
                if self.iteration_count >= self.max_iterations:
                    break

                self._research_knowledge_gap(target)

            # Form conclusions from current evidence
            self._form_conclusions()

            # Update session
            self.current_session.updated_at = datetime.utcnow()
            self.current_session.total_iterations = self.iteration_count
            self.current_session.reasoning_summary = (
                self.reasoning_engine.get_reasoning_summary()
            )
            self.current_session.learning_stats = (
                self.continuous_learner.get_learning_stats()
            )

            # Check if we have enough information
            if self._has_sufficient_information():
                break

        # Generate final answer
        final_answer = self._generate_final_answer()
        self.current_session.final_answer = final_answer
        self.current_session.status = "completed"
        self.current_session.updated_at = datetime.utcnow()

        # Export results
        self._export_research_results()

        print("\n✅ RESEARCH COMPLETED")
        print(f"   Iterations: {self.iteration_count}")
        print(f"   Sources: {len(self.current_session.sources_used)}")
        print(f"   Conclusions: {len(self.current_session.conclusions)}")
        print(f"   Learning updates: {self.continuous_learner.update_count}")

        return self.current_session

    def _generate_initial_queries(self, question: str) -> list[str]:
        """Generate initial search queries for the research question."""
        query_prompt = f"""<query_generation>
<research_question>
{question}
</research_question>
<task>Generate 5 diverse search queries to research this question</task>
<requirements>
1. Specific and focused
2. Cover different aspects of the question
3. Include technical terms if relevant
4. Vary in specificity (broad to narrow)
</requirements>
<output_format>
<queries>
  <query>specific search query 1</query>
  <query>specific search query 2</query>
  <query>specific search query 3</query>
  <query>specific search query 4</query>
  <query>specific search query 5</query>
</queries>
</output_format>
</query_generation>"""

        try:
            # Use centralized aggregator when available
            if getattr(self, "aggregator", None):
                response = self.aggregator.generate(query_prompt)
            else:
                response = self.ai_generator.generate_direct_response(query_prompt)

            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response)

            queries = []
            for query_elem in root.findall(".//query"):
                queries.append(query_elem.text.strip())

            # Log the initial queries
            self.reasoning_engine.log_thought(
                "decision",
                f"Generated initial search queries: {queries}",
                confidence=0.8,
            )

            # Create knowledge gaps for each query
            for query in queries:
                self.reasoning_engine.identify_knowledge_gap(query, [])

            return queries

        except Exception as e:
            print(f"Failed to generate initial queries: {e}")
            # Fallback queries
            fallback_queries = [
                question,
                f"research {question}",
                f"information about {question}",
                f"analysis of {question}",
                f"latest {question}",
            ]

            for query in fallback_queries:
                self.reasoning_engine.identify_knowledge_gap(query, [])

            return fallback_queries

    def _research_knowledge_gap(self, gap):
        """Research a specific knowledge gap using vision-based automation."""
        print(f"\n🔍 RESEARCHING: {gap.question}")

        # Start a vision session for this research goal
        self.browser.start_vision_session(user_goal=f"Research: {gap.question}", initial_url="https://www.google.com")

        try:
            # Use the gap's search queries for intelligent navigation
            for query in gap.search_queries[:2]:  # Use first 2 queries
                print(f"   Researching: {query}")

                # Navigate to search engine and perform intelligent search
                self._perform_vision_search(query, gap)

        finally:
            # End the vision session
            session = self.browser.end_vision_session()
            print(
                f"Vision session completed with {len(session.actions_performed)} actions"
            )

    def _perform_vision_search(self, query: str, gap):
        """Perform intelligent search using vision-based automation."""
        try:
            # Navigate to Google search
            self.browser.navigate_to("https://www.google.com")

            # Analyze the page with vision to understand the layout
            vision_analysis = self.browser.analyze_page_with_vision(
                user_goal=f"Search for: {query}"
            )

            if vision_analysis and vision_analysis.recommended_actions:
                # Execute recommended actions (like clicking search box, typing query, etc.)
                for action in vision_analysis.recommended_actions:
                    if action.confidence > 0.7:  # Only execute high-confidence actions
                        print(
                            f"   Executing action: {action.action_type} - {action.target_description}"
                        )
                        success = self.browser.execute_vision_action(action)

                        if success:
                            # Wait for page to load
                            time.sleep(2)

                            # Analyze results page
                            results_analysis = self.browser.analyze_page_with_vision(
                                user_goal=f"Find relevant results for: {query}"
                            )

                            if results_analysis:
                                # Process search results intelligently
                                self._process_search_results(results_analysis, gap)

                        # Limit actions to avoid infinite loops
                        if len(self.browser.vision_actions) > 10:
                            break
            else:
                print(f"   No vision actions recommended for query: {query}")

        except Exception as e:
            print(f"   Vision search failed for {query}: {e}")

    def _process_search_results(self, vision_analysis, gap):
        """Process search results using vision analysis."""
        if not vision_analysis.recommended_actions:
            return

        # Look for link actions to follow
        for action in vision_analysis.recommended_actions:
            if action.action_type == "navigate" and action.confidence > 0.6:
                print(f"   Following link: {action.target_description}")

                # Execute the navigation action
                success = self.browser.execute_vision_action(action)

                if success:
                    # Wait for page to load
                    time.sleep(3)

                    # Analyze the new page
                    page_analysis = self.browser.analyze_page_with_vision(
                        user_goal=f"Extract information about: {gap.question}"
                    )

                    if page_analysis:
                        # Extract content from the page
                        self._extract_page_content(page_analysis, gap)

                    # Go back to search results
                    self.browser.go_back()
                    time.sleep(2)

    def _extract_page_content(self, page_analysis, gap):
        """Extract and process content from a page using vision analysis."""
        try:
            # Get current page content
            current_url = self.browser.page.url if self.browser.page else ""

            if current_url and current_url not in self.current_session.sources_used:
                # Get page content
                page_content = self.browser.page.content() if self.browser.page else ""

                if page_content:
                    # Parse the content
                    parsed_content = self.parser.parse(
                        current_url, page_content, "text/html"
                    )

                    if parsed_content:
                        # Analyze with reasoning engine
                        thought_id = self.reasoning_engine.analyze_content(
                            parsed_content.content, current_url
                        )

                        # Add learning example
                        self.continuous_learner.add_learning_example(
                            input_text=f"Research question: {gap.question}",
                            output_text=parsed_content.content[:500],  # First 500 chars
                            confidence=0.7,
                            source_url=current_url,
                            thought_id=thought_id,
                        )

                        # Add to sources used
                        self.current_session.sources_used.append(current_url)

                        print(f"   Extracted content from: {current_url}")

        except Exception as e:
            print(f"   Failed to extract content: {e}")

    def _fetch_and_analyze_url(self, url: str, gap):
        """Fetch and analyze content from a URL."""
        print(f"   Fetching: {url}")

        try:
            # Fetch content
            result = self.browser.fetch(url)
            if not result:
                return

            # Parse content
            parsed_content = self.parser.parse(
                result.url, result.content, result.mime_type
            )
            if not parsed_content:
                return

            # Analyze with reasoning engine
            thought_id = self.reasoning_engine.analyze_content(
                parsed_content.content, url
            )

            # Add learning example
            self.continuous_learner.add_learning_example(
                input_text=f"Research question: {gap.question}",
                output_text=parsed_content.content[:500],  # First 500 chars
                confidence=0.7,
                source_url=url,
                thought_id=thought_id,
            )

            # Analyze images if present
            if result.screenshot_path:
                self._analyze_image(result.screenshot_path, url)

        except Exception as e:
            print(f"   Failed to analyze {url}: {e}")

    def _analyze_image(self, image_path: str, source_url: str):
        """Analyze image content for additional insights."""
        try:
            analysis = self.image_analyzer.analyze_image(image_path)
            if analysis:
                # ImageAnalysis has no 'description' field; use generated page description when available
                desc = getattr(analysis, "content_type", None) or "image analysis"
                self.reasoning_engine.log_thought(
                    "observation",
                    f"Image analysis: {desc}",
                    confidence=0.6,
                    evidence=[source_url],
                )
        except Exception as e:
            print(f"   Image analysis failed: {e}")

    def _form_conclusions(self):
        """Form conclusions from current evidence."""
        print("\n🤔 FORMING CONCLUSIONS")

        # Get current hypotheses
        hypotheses = self.reasoning_engine.working_memory.hypotheses

        for hypothesis in hypotheses:
            # Get evidence for this hypothesis
            evidence = []
            for thought in self.reasoning_engine.thoughts:
                if hypothesis.lower() in thought.content.lower():
                    evidence.extend(thought.evidence)

            if evidence:
                # Form conclusion
                conclusion_id = self.reasoning_engine.form_conclusion(
                    hypothesis, evidence
                )

                # Add to session conclusions
                conclusion_thought = next(
                    (
                        t
                        for t in self.reasoning_engine.thoughts
                        if t.id == conclusion_id
                    ),
                    None,
                )
                if conclusion_thought:
                    self.current_session.conclusions.append(conclusion_thought.content)

    def _has_sufficient_information(self) -> bool:
        """Check if we have sufficient information to answer the question."""
        # Check if we have enough conclusions
        if len(self.current_session.conclusions) >= 3:
            return True

        # Check if we have enough sources
        if len(self.current_session.sources_used) >= 10:
            return True

        # Check if reasoning engine thinks we're done
        summary = self.reasoning_engine.get_reasoning_summary()
        if summary["knowledge_gaps"]["filled"] > summary["knowledge_gaps"]["unfilled"]:
            return True

        return False

    def _generate_final_answer(self) -> str:
        """Generate a comprehensive final answer."""
        print("\n📝 GENERATING FINAL ANSWER")

        # Prepare context for answer generation (constructed inline when needed)

        answer_prompt = f"""
        Based on the research conducted, provide a comprehensive answer to: {self.current_session.question}
        
        Context:
        - Conclusions reached: {self.current_session.conclusions}
        - Sources consulted: {len(self.current_session.sources_used)} sources
        - Reasoning steps: {self.current_session.reasoning_summary["total_thoughts"]} thoughts
        - Learning updates: {self.current_session.learning_stats["update_count"]} model updates
        
        Provide a well-structured answer that:
        1. Directly addresses the question
        2. Summarizes key findings
        3. Acknowledges uncertainties
        4. Cites sources where appropriate
        5. Shows the reasoning process
        """

        try:
            if getattr(self, "aggregator", None):
                final_answer = self.aggregator.generate(answer_prompt)
            else:
                final_answer = self.ai_generator.generate_response(answer_prompt)

            # Log the final answer
            self.reasoning_engine.log_thought(
                "conclusion",
                f"Final answer generated: {final_answer[:200]}...",
                confidence=0.9,
                evidence=self.current_session.sources_used,
            )

            return final_answer

        except Exception as e:
            print(f"Failed to generate final answer: {e}")
            return f"Research completed but failed to generate final answer: {e}"

    def _export_research_results(self):
        """Export complete research results."""
        try:
            # Export reasoning trace
            reasoning_trace = self.reasoning_engine.export_reasoning_trace()

            # Export learning trace
            learning_trace = self.continuous_learner.export_learning_trace()

            # Combine into research report
            research_report = {
                "research_session": self.current_session.model_dump(),
                "reasoning_trace": reasoning_trace,
                "learning_trace": learning_trace,
                "exported_at": datetime.utcnow().isoformat(),
            }

            # Save to file
            report_path = Path(f"reports/research_{self.current_session.id}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(research_report, f, indent=2, default=str)

            print(f"Research report saved: {report_path}")

        except Exception as e:
            print(f"Failed to export research results: {e}")

    def get_research_stats(self) -> dict[str, Any]:
        """Get statistics about the current research session."""
        if not self.current_session:
            return {}

        return {
            "session_id": self.current_session.id,
            "question": self.current_session.question,
            "status": self.current_session.status,
            "iterations": self.iteration_count,
            "sources_used": len(self.current_session.sources_used),
            "conclusions": len(self.current_session.conclusions),
            "reasoning_summary": self.current_session.reasoning_summary,
            "learning_stats": self.current_session.learning_stats,
        }

    def close(self):
        """Close browser and cleanup resources."""
        if hasattr(self, "browser"):
            self.browser.close()

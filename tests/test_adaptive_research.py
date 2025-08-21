"""Tests for adaptive research components."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_infant.core.adaptive_loop import AdaptiveResearchLoop, ResearchSession
from ai_infant.core.reasoning import (
    KnowledgeGap,
    ReasoningEngine,
)
from ai_infant.data import Store
from ai_infant.learn.continuous import ContinuousLearner


class TestReasoningEngine:
    """Test the reasoning engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = Mock()
        self.ai_generator = Mock()
        self.reasoning_engine = ReasoningEngine(self.store, self.ai_generator)

    def test_log_thought(self):
        """Test logging a thought."""
        thought_id = self.reasoning_engine.log_thought(
            "observation",
            "Test observation",
            confidence=0.8,
            evidence=["http://example.com"],
        )

        assert thought_id is not None
        assert len(self.reasoning_engine.thoughts) == 1

        thought = self.reasoning_engine.thoughts[0]
        assert thought.thought_type == "observation"
        assert thought.content == "Test observation"
        assert thought.confidence == 0.8
        assert thought.evidence == ["http://example.com"]

    def test_identify_knowledge_gap(self):
        """Test identifying a knowledge gap."""
        gap_id = self.reasoning_engine.identify_knowledge_gap(
            "What is quantum computing?", ["thought_1", "thought_2"]
        )

        assert gap_id is not None
        assert len(self.reasoning_engine.knowledge_gaps) == 1

        gap = self.reasoning_engine.knowledge_gaps[0]
        assert gap.question == "What is quantum computing?"
        assert gap.related_thoughts == ["thought_1", "thought_2"]
        assert gap.importance == 0.5  # Base importance for this question
        assert not gap.filled

    def test_get_next_search_targets(self):
        """Test getting next search targets."""
        # Add some knowledge gaps
        self.reasoning_engine.identify_knowledge_gap("Question 1", [])
        self.reasoning_engine.identify_knowledge_gap("Question 2", [])
        self.reasoning_engine.identify_knowledge_gap("Question 3", [])

        targets = self.reasoning_engine.get_next_search_targets(max_targets=2)
        assert len(targets) == 2
        assert all(isinstance(target, KnowledgeGap) for target in targets)

    def test_get_reasoning_summary(self):
        """Test getting reasoning summary."""
        # Add some thoughts
        self.reasoning_engine.log_thought("observation", "Test 1")
        self.reasoning_engine.log_thought("hypothesis", "Test 2")
        self.reasoning_engine.log_thought("conclusion", "Test 3")

        summary = self.reasoning_engine.get_reasoning_summary()

        assert summary["total_thoughts"] == 3
        assert summary["thought_types"]["observation"] == 1
        assert summary["thought_types"]["hypothesis"] == 1
        assert summary["thought_types"]["conclusion"] == 1


class TestContinuousLearner:
    """Test the continuous learning engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = Mock()
        self.learner = ContinuousLearner(self.store)

    def test_add_learning_example(self):
        """Test adding a learning example."""
        self.learner.add_learning_example(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            confidence=0.9,
            source_url="http://example.com",
            thought_id="thought_1",
        )

        assert len(self.learner.memory_buffer) == 1

        example = self.learner.memory_buffer[0]
        assert example["input_text"] == "What is AI?"
        assert example["output_text"] == "AI is artificial intelligence"
        assert example["confidence"] == 0.9
        assert example["source_url"] == "http://example.com"
        assert example["thought_id"] == "thought_1"
        assert not example["used_for_training"]

    def test_get_learning_stats(self):
        """Test getting learning statistics."""
        # Add some examples
        self.learner.add_learning_example(
            "Input 1", "Output 1", 0.8, "url1", "thought1"
        )
        self.learner.add_learning_example(
            "Input 2", "Output 2", 0.9, "url2", "thought2"
        )

        stats = self.learner.get_learning_stats()

        assert stats["buffer_size"] == 2
        assert (
            stats["high_confidence_examples"] == 2
        )  # Both 0.8 and 0.9 are high confidence
        assert stats["update_count"] == 0
        assert not stats["model_loaded"]  # No model loaded in test


class TestAdaptiveResearchLoop:
    """Test the adaptive research loop."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = Mock()
        with patch("ai_infant.crawl.browser.sync_playwright"):
            self.loop = AdaptiveResearchLoop(self.store, headless=True)

    def test_research_session_creation(self):
        """Test creating a research session."""
        session = ResearchSession(
            id="test_id",
            question="What is AI?",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            status="active",
            reasoning_summary={},
            learning_stats={},
            conclusions=[],
            sources_used=[],
            total_iterations=0,
        )

        assert session.id == "test_id"
        assert session.question == "What is AI?"
        assert session.status == "active"
        assert len(session.conclusions) == 0
        assert len(session.sources_used) == 0

    def test_get_research_stats_empty(self):
        """Test getting research stats when no session exists."""
        stats = self.loop.get_research_stats()
        assert stats == {}

    @patch(
        "ai_infant.core.adaptive_loop.AdaptiveResearchLoop._generate_initial_queries"
    )
    @patch("ai_infant.core.adaptive_loop.AdaptiveResearchLoop._research_knowledge_gap")
    @patch("ai_infant.core.adaptive_loop.AdaptiveResearchLoop._form_conclusions")
    @patch(
        "ai_infant.core.adaptive_loop.AdaptiveResearchLoop._has_sufficient_information"
    )
    @patch("ai_infant.core.adaptive_loop.AdaptiveResearchLoop._generate_final_answer")
    def test_research_question_mock(
        self,
        mock_final_answer,
        mock_sufficient,
        mock_conclusions,
        mock_research,
        mock_queries,
    ):
        """Test research question with mocked components."""
        # Mock the AI generator
        self.loop.ai_generator.generate_response.return_value = '["query1", "query2"]'

        # Mock return values
        mock_queries.return_value = ["query1", "query2"]
        mock_sufficient.return_value = True
        mock_final_answer.return_value = "Final answer"

        # Mock reasoning engine to return empty search targets
        self.loop.reasoning_engine.get_next_search_targets.return_value = []

        session = self.loop.research_question("What is AI?")

        assert session.question == "What is AI?"
        assert session.status == "completed"
        assert session.final_answer == "Final answer"
        assert session.total_iterations == 1


class TestIntegration:
    """Integration tests for the adaptive research system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = Store(str(self.db_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_reasoning_engine_with_store(self):
        """Test reasoning engine with actual store."""
        ai_generator = Mock()
        ai_generator.generate_response.return_value = '["query1", "query2"]'

        reasoning_engine = ReasoningEngine(self.store, ai_generator)

        # Log a thought
        thought_id = reasoning_engine.log_thought(
            "observation", "Test observation", confidence=0.8
        )

        assert thought_id is not None

        # Check that trace was stored
        # Note: In a real test, we'd verify the trace was stored in the database

    def test_continuous_learner_with_store(self):
        """Test continuous learner with actual store."""
        learner = ContinuousLearner(self.store)

        # Add learning example
        learner.add_learning_example(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            confidence=0.9,
            source_url="http://example.com",
            thought_id="thought_1",
        )

        stats = learner.get_learning_stats()
        assert stats["buffer_size"] == 1
        assert stats["high_confidence_examples"] == 1


if __name__ == "__main__":
    pytest.main([__file__])

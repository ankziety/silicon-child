"""Tests for LLM Jury evaluation and promotion system."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai_infant.data.store import Store
from ai_infant.learn.eval import (
    EvaluationError,
    EvaluationResult,
    JuryResult,
    LLMJury,
    GPT4oMiniJudge,
    GPT5Judge,
    ClaudeHaikuJudge,
    ClaudeSonnetJudge,
    CommandRPlusJudge,
    create_frontier_jury,
    create_diverse_jury,
    create_affordable_jury,
    create_specialized_jury,
    create_mixed_jury,
)
from scripts.promote import AdapterInfo, PromotionError, PromotionManager


class TestLLMJudges:
    """Test individual LLM judges."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_gpt4o_mini_judge_initialization(self):
        """Test GPT-4o-mini judge initialization."""
        judge = GPT4oMiniJudge("test_judge", "general")
        assert judge.name == "test_judge"
        assert judge.evaluation_type == "general"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_gpt5_judge_initialization(self):
        """Test GPT-5 judge initialization."""
        judge = GPT5Judge("test_judge", "general")
        assert judge.name == "test_judge"
        assert judge.evaluation_type == "general"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_claude_haiku_judge_initialization(self):
        """Test Claude Haiku judge initialization."""
        judge = ClaudeHaikuJudge("test_judge", "general")
        assert judge.name == "test_judge"
        assert judge.evaluation_type == "general"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_claude_sonnet_judge_initialization(self):
        """Test Claude Sonnet judge initialization."""
        judge = ClaudeSonnetJudge("test_judge", "general")
        assert judge.name == "test_judge"
        assert judge.evaluation_type == "general"
    
    @patch.dict(os.environ, {"COHERE_API_KEY": "test_key"})
    def test_commandr_plus_judge_initialization(self):
        """Test Command R+ judge initialization."""
        judge = CommandRPlusJudge("test_judge", "general")
        assert judge.name == "test_judge"
        assert judge.evaluation_type == "general"
    
    def test_judge_missing_api_key(self):
        """Test that judges fail when API keys are missing."""
        with pytest.raises(EvaluationError, match="OPENAI_API_KEY environment variable not set"):
            GPT4oMiniJudge("test_judge", "general")
        
        with pytest.raises(EvaluationError, match="OPENAI_API_KEY environment variable not set"):
            GPT5Judge("test_judge", "general")
        
        with pytest.raises(EvaluationError, match="ANTHROPIC_API_KEY environment variable not set"):
            ClaudeHaikuJudge("test_judge", "general")
        
        with pytest.raises(EvaluationError, match="ANTHROPIC_API_KEY environment variable not set"):
            ClaudeSonnetJudge("test_judge", "general")
        
        with pytest.raises(EvaluationError, match="COHERE_API_KEY environment variable not set"):
            CommandRPlusJudge("test_judge", "general")


class TestLLMJury:
    """Test LLM jury system."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key", 
        "COHERE_API_KEY": "test_key"
    })
    def test_jury_initialization(self):
        """Test jury initialization with valid judges."""
        judges = [
            GPT4oMiniJudge("gpt4o_mini_judge", "general"),
            ClaudeHaikuJudge("claude_haiku_judge", "general"),
            CommandRPlusJudge("commandr_plus_judge", "general")
        ]
        jury = LLMJury(judges, aggregation_method="average")
        assert len(jury.judges) == 3
        assert jury.aggregation_method == "average"
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key"
    })
    def test_jury_initialization_insufficient_judges(self):
        """Test jury initialization with insufficient judges."""
        judges = [GPT4oMiniJudge("gpt4o_mini_judge", "general"), ClaudeHaikuJudge("claude_haiku_judge", "general")]
        with pytest.raises(ValueError, match="at least 3 judges"):
            LLMJury(judges)
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_jury_aggregation_methods(self):
        """Test different aggregation methods."""
        judges = [
            GPT4oMiniJudge("gpt4o_mini_judge", "general"),
            ClaudeHaikuJudge("claude_haiku_judge", "general"),
            CommandRPlusJudge("commandr_plus_judge", "general")
        ]
        
        # Test average aggregation
        jury = LLMJury(judges, aggregation_method="average")
        assert jury.aggregation_method == "average"
        
        # Test median aggregation
        jury = LLMJury(judges, aggregation_method="median")
        assert jury.aggregation_method == "median"
        
        # Test weighted average aggregation
        jury = LLMJury(judges, aggregation_method="weighted_average")
        assert jury.aggregation_method == "weighted_average"
        
        # Test invalid aggregation method
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            jury = LLMJury(judges, aggregation_method="invalid")
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_create_jury_functions(self):
        """Test jury creation functions."""
        # Test frontier jury
        frontier_jury = create_frontier_jury()
        assert len(frontier_jury.judges) == 5
        assert frontier_jury.aggregation_method == "average"
        
        # Test diverse jury
        diverse_jury = create_diverse_jury()
        assert len(diverse_jury.judges) == 3
        assert diverse_jury.aggregation_method == "median"
        
        # Test specialized jury
        specialized_jury = create_specialized_jury()
        assert len(specialized_jury.judges) == 4
        assert specialized_jury.aggregation_method == "weighted_average"


class TestAdapterInfo:
    """Test adapter information class."""
    
    def test_adapter_info_creation(self):
        """Test adapter info creation and serialization."""
        from datetime import datetime
        timestamp = datetime.utcnow()
        adapter = AdapterInfo("model1", 0.85, timestamp)
        
        assert adapter.model_path == "model1"
        assert adapter.score == 0.85
        assert adapter.timestamp == timestamp
    
    def test_adapter_info_serialization(self):
        """Test adapter info serialization to dict."""
        from datetime import datetime
        timestamp = datetime.utcnow()
        adapter = AdapterInfo("model1", 0.85, timestamp)
        
        data = adapter.to_dict()
        assert data["model_path"] == "model1"
        assert data["score"] == 0.85
        assert "timestamp" in data
    
    def test_adapter_info_deserialization(self):
        """Test adapter info deserialization from dict."""
        from datetime import datetime
        timestamp = datetime.utcnow()
        original = AdapterInfo("model1", 0.85, timestamp)
        
        data = original.to_dict()
        restored = AdapterInfo.from_dict(data)
        
        assert restored.model_path == original.model_path
        assert restored.score == original.score


class TestPromotionManager:
    """Test promotion manager."""
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary store for testing."""
        # Create a temporary directory for the database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Initialize the store with a fresh database
            store = Store(str(db_path))
            yield store
    
    @pytest.fixture
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def promotion_manager(self, temp_store):
        """Create promotion manager for testing."""
        jury = create_diverse_jury()
        return PromotionManager(temp_store, jury, max_adapters=3)
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_promotion_manager_initialization(self, temp_store):
        """Test promotion manager initialization."""
        jury = create_diverse_jury()
        manager = PromotionManager(temp_store, jury)
        
        assert len(manager.adapters) == 0
        assert manager.max_adapters == 5  # default
    
    def test_get_incumbent_score_empty(self, promotion_manager):
        """Test getting incumbent score with no adapters."""
        score = promotion_manager.get_incumbent_score()
        assert score == 0.0
    
    def test_get_incumbent_path_empty(self, promotion_manager):
        """Test getting incumbent path with no adapters."""
        path = promotion_manager.get_incumbent_path()
        assert path is None
    
    def test_promote_candidate_first_adapter(self, promotion_manager):
        """Test promoting first candidate (should always succeed)."""
        result = promotion_manager.promote_candidate(
            model_path="model1",
            prompt="Test prompt",
            response="Test response",
            context="Test context",
            seed=42
        )
        
        assert result["promoted"] is True
        assert result["candidate_score"] > 0.0
        assert result["incumbent_score"] == 0.0
        assert len(promotion_manager.adapters) == 1
    
    def test_promote_candidate_better_score(self, promotion_manager):
        """Test promoting candidate with better score."""
        # Add first adapter
        promotion_manager.promote_candidate(
            model_path="model1",
            prompt="Test prompt",
            response="Test response",
            context="Test context",
            seed=42
        )
        
        # Mock jury to return higher score for second model
        with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.candidate_score = 0.9
            mock_result.judge_results = []
            mock_result.aggregation_method = "average"
            mock_result.seed = 43
            mock_evaluate.return_value = mock_result
            
            result = promotion_manager.promote_candidate(
                model_path="model2",
                prompt="Test prompt",
                response="Test response",
                context="Test context",
                seed=43
            )
            
            assert result["promoted"] is True
            assert result["candidate_score"] == 0.9
            assert len(promotion_manager.adapters) == 2
    
    def test_promote_candidate_worse_score(self, promotion_manager):
        """Test rejecting candidate with worse score."""
        # Add first adapter with high score
        with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.candidate_score = 0.9
            mock_result.judge_results = []
            mock_result.aggregation_method = "average"
            mock_result.seed = 42
            mock_evaluate.return_value = mock_result
            
            promotion_manager.promote_candidate(
                model_path="model1",
                prompt="Test prompt",
                response="Test response",
                context="Test context",
                seed=42
            )
        
        # Try to promote worse candidate
        with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.candidate_score = 0.3
            mock_result.judge_results = []
            mock_result.aggregation_method = "average"
            mock_result.seed = 43
            mock_evaluate.return_value = mock_result
            
            result = promotion_manager.promote_candidate(
                model_path="model2",
                prompt="Test prompt",
                response="Test response",
                context="Test context",
                seed=43
            )
            
            assert result["promoted"] is False
            assert result["candidate_score"] == 0.3
            assert result["incumbent_score"] == 0.9
            assert len(promotion_manager.adapters) == 1  # No new adapter added
    
    def test_ring_buffer_overflow(self, promotion_manager):
        """Test ring buffer behavior when max adapters reached."""
        # Add 3 adapters (max_adapters=3)
        for i in range(3):
            with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
                mock_result = Mock()
                mock_result.candidate_score = 0.5 + i * 0.1
                mock_result.judge_results = []
                mock_result.aggregation_method = "average"
                mock_result.seed = 42 + i
                mock_evaluate.return_value = mock_result
                
                promotion_manager.promote_candidate(
                    model_path=f"model{i}",
                    prompt="Test prompt",
                    response="Test response",
                    context="Test context",
                    seed=42 + i
                )
        
        assert len(promotion_manager.adapters) == 3
        
        # Add one more adapter - should remove oldest
        with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.candidate_score = 0.8
            mock_result.judge_results = []
            mock_result.aggregation_method = "average"
            mock_result.seed = 45
            mock_evaluate.return_value = mock_result
            
            promotion_manager.promote_candidate(
                model_path="model4",
                prompt="Test prompt",
                response="Test response",
                context="Test context",
                seed=45
            )
        
        assert len(promotion_manager.adapters) == 3  # Still max_adapters
        assert promotion_manager.adapters[0].model_path == "model1"  # Oldest removed
        assert promotion_manager.adapters[-1].model_path == "model4"  # Newest added
    
    def test_get_adapter_history(self, promotion_manager):
        """Test getting adapter history."""
        # Add some adapters
        with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.candidate_score = 0.5
            mock_result.judge_results = []
            mock_result.aggregation_method = "average"
            mock_result.seed = 42
            mock_evaluate.return_value = mock_result
            
            promotion_manager.promote_candidate(
                model_path="model1",
                prompt="Test prompt",
                response="Test response",
                context="Test context",
                seed=42
            )
        
        history = promotion_manager.get_adapter_history()
        assert len(history) == 1
        assert history[0]["model_path"] == "model1"
        assert history[0]["score"] == 0.5
    
    def test_rollback_to_adapter(self, promotion_manager):
        """Test rollback functionality."""
        # Add multiple adapters
        for i in range(3):
            with patch.object(promotion_manager.jury, 'evaluate') as mock_evaluate:
                mock_result = Mock()
                mock_result.candidate_score = 0.5 + i * 0.1
                mock_result.judge_results = []
                mock_result.aggregation_method = "average"
                mock_result.seed = 42 + i
                mock_evaluate.return_value = mock_result
                
                promotion_manager.promote_candidate(
                    model_path=f"model{i}",
                    prompt="Test prompt",
                    response="Test response",
                    context="Test context",
                    seed=42 + i
                )
        
        assert len(promotion_manager.adapters) == 3
        
        # Rollback to middle adapter
        success = promotion_manager.rollback_to_adapter("model1")
        assert success is True
        assert len(promotion_manager.adapters) == 2
        assert promotion_manager.adapters[-1].model_path == "model1"
    
    def test_rollback_to_nonexistent_adapter(self, promotion_manager):
        """Test rollback to non-existent adapter."""
        success = promotion_manager.rollback_to_adapter("nonexistent")
        assert success is False
    
    def test_promote_candidate_evaluation_failure(self, promotion_manager):
        """Test that promotion fails when evaluation fails."""
        with pytest.raises(PromotionError, match="Evaluation failed"):
            promotion_manager.promote_candidate(
                model_path="model1",
                prompt="Test prompt",
                response="",  # Empty response should cause issues
                context="Test context",
                seed=42
            )


class TestJobLogging:
    """Test job logging functionality."""
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary store for testing."""
        # Create a temporary directory for the database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Initialize the store with a fresh database
            store = Store(str(db_path))
            yield store
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_evaluation_job_logging(self, temp_store):
        """Test that evaluation jobs are properly logged."""
        jury = create_diverse_jury()
        manager = PromotionManager(temp_store, jury)
        
        # Run evaluation
        result = manager.evaluate_candidate(
            prompt="Test prompt",
            response="Test response",
            context="Test context",
            seed=42
        )
        
        # Check that job was logged
        jobs = temp_store.conn.execute("SELECT * FROM jobs WHERE type = 'eval'").fetchall()
        assert len(jobs) == 1
        
        job = jobs[0]
        assert job[2] == "completed"  # status
        
        # Check input data
        input_data = json.loads(job[8])  # input_data
        assert input_data["prompt"] == "Test prompt"
        assert input_data["response"] == "Test response"
        assert input_data["context"] == "Test context"
        assert input_data["seed"] == 42
        
        # Check output data
        output_data = json.loads(job[9])  # output_data
        assert output_data["candidate_score"] == result.candidate_score
        assert "judge_results" in output_data
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_promotion_job_logging(self, temp_store):
        """Test that promotion jobs are properly logged."""
        jury = create_diverse_jury()
        manager = PromotionManager(temp_store, jury)
        
        # Run promotion
        result = manager.promote_candidate(
            model_path="model1",
            prompt="Test prompt",
            response="Test response",
            context="Test context",
            seed=42
        )
        
        # Check that jobs were logged
        eval_jobs = temp_store.conn.execute("SELECT * FROM jobs WHERE type = 'eval'").fetchall()
        promote_jobs = temp_store.conn.execute("SELECT * FROM jobs WHERE type = 'promote'").fetchall()
        
        assert len(eval_jobs) == 1
        assert len(promote_jobs) == 1
        
        # Check promotion job
        promote_job = promote_jobs[0]
        assert promote_job[2] == "completed"  # status
        
        # Check input data
        input_data = json.loads(promote_job[8])  # input_data
        assert input_data["model_path"] == "model1"
        assert input_data["prompt"] == "Test prompt"
        assert input_data["response"] == "Test response"
        assert input_data["context"] == "Test context"
        assert input_data["seed"] == 42
        assert "candidate_score" in input_data
        assert "incumbent_score" in input_data
        
        # Check output data
        output_data = json.loads(promote_job[9])  # output_data
        assert output_data["promoted"] == result["promoted"]
        assert "jury_result" in output_data
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "COHERE_API_KEY": "test_key"
    })
    def test_failed_evaluation_job_logging(self, temp_store):
        """Test that failed evaluations are properly logged."""
        jury = create_diverse_jury()
        manager = PromotionManager(temp_store, jury)
        
        # Try to run evaluation that will fail
        with pytest.raises(PromotionError):
            manager.evaluate_candidate(
                prompt="Test prompt",
                response="",  # Empty response should cause issues
                context="Test context",
                seed=42
            )
        
        # Check that failed job was logged
        jobs = temp_store.conn.execute("SELECT * FROM jobs WHERE type = 'eval'").fetchall()
        assert len(jobs) == 1
        
        job = jobs[0]
        assert job[2] == "failed"  # status
        
        # Check error data
        error_data = json.loads(job[10])  # error_data
        assert error_data["type"] == "evaluation_error"
        assert "evaluation failed" in error_data["message"].lower()

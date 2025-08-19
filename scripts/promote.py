"""Promotion system for model adapters with ring buffer rollback."""

import json
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_infant.learn.eval import LLMJury, JuryResult, EvaluationError
from ai_infant.data.store import Store


class AdapterInfo:
    """Information about a model adapter."""
    
    def __init__(self, model_path: str, score: float, timestamp: datetime):
        self.model_path = model_path
        self.score = score
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_path": self.model_path,
            "score": self.score,
            "timestamp": self.timestamp.isoformat() + "Z"
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterInfo":
        """Create from dictionary."""
        return cls(
            model_path=data["model_path"],
            score=data["score"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        )


class PromotionError(Exception):
    """Raised when promotion cannot be performed."""
    pass


class PromotionManager:
    """Manages model adapter promotion with ring buffer rollback."""
    
    def __init__(self, store: Store, jury: LLMJury, max_adapters: int = 5):
        """Initialize promotion manager."""
        self.store = store
        self.jury = jury
        self.max_adapters = max_adapters
        self.adapters: deque = deque(maxlen=max_adapters)
        self._load_existing_adapters()
    
    def _load_existing_adapters(self) -> None:
        """Load existing adapters from storage."""
        adapters_file = Path("data/adapters.json")
        if adapters_file.exists():
            with open(adapters_file, "r") as f:
                data = json.load(f)
                for adapter_data in data.get("adapters", []):
                    adapter = AdapterInfo.from_dict(adapter_data)
                    self.adapters.append(adapter)
    
    def _save_adapters(self) -> None:
        """Save adapters to storage."""
        adapters_file = Path("data/adapters.json")
        adapters_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "adapters": [adapter.to_dict() for adapter in self.adapters],
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(adapters_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _log_job(
        self,
        job_type: str,
        status: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        error_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a job entry."""
        job_id = f"{job_type}-{int(time.time() * 1000)}"
        now = datetime.utcnow().isoformat() + "Z"
        
        job_data = {
            "id": job_id,
            "type": job_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "input": input_data,
            "output": output_data,
            "error": error_data,
            "metadata": {
                "version": "0.1.0",
                "priority": 5
            }
        }
        
        self.store.store_job(job_data)
        return job_id
    
    def get_incumbent_score(self) -> float:
        """Get the score of the current incumbent (best adapter)."""
        if not self.adapters:
            return 0.0
        
        # Return the highest score among current adapters
        return max(adapter.score for adapter in self.adapters)
    
    def get_incumbent_path(self) -> Optional[str]:
        """Get the path of the current incumbent adapter."""
        if not self.adapters:
            return None
        
        # Return the path of the adapter with the highest score
        best_adapter = max(self.adapters, key=lambda a: a.score)
        return best_adapter.model_path
    
    def evaluate_candidate(
        self, 
        prompt: str,
        response: str,
        context: Optional[str] = None,
        seed: Optional[int] = None
    ) -> JuryResult:
        """Evaluate a candidate response using the LLM jury."""
        # Log evaluation start
        input_data = {
            "prompt": prompt,
            "response": response,
            "context": context,
            "seed": seed,
            "judge_count": len(self.jury.judges),
            "aggregation_method": self.jury.aggregation_method
        }
        
        job_id = self._log_job("eval", "running", input_data)
        
        try:
            # Run evaluation
            result = self.jury.evaluate(prompt, response, context, seed)
            
            # Log successful evaluation
            output_data = {
                "candidate_score": result.candidate_score,
                "judge_results": [
                    {
                        "name": r.judge_name,
                        "score": r.score,
                        "reasoning": r.reasoning,
                        "metadata": r.metadata
                    }
                    for r in result.judge_results
                ],
                "aggregation_method": result.aggregation_method,
                "seed": result.seed,
                "metadata": result.metadata
            }
            
            self._log_job("eval", "completed", input_data, output_data)
            
            return result
            
        except Exception as e:
            # Log failed evaluation
            error_data = {
                "type": "evaluation_error",
                "message": str(e),
                "stack": None
            }
            
            self._log_job("eval", "failed", input_data, error_data=error_data)
            raise PromotionError(f"Evaluation failed: {e}") from e
    
    def promote_candidate(
        self, 
        model_path: str,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate and potentially promote a candidate model."""
        # Get current incumbent score
        incumbent_score = self.get_incumbent_score()
        
        # Evaluate candidate
        jury_result = self.evaluate_candidate(prompt, response, context, seed)
        candidate_score = jury_result.candidate_score
        
        # Log promotion decision start
        input_data = {
            "model_path": model_path,
            "prompt": prompt,
            "response": response,
            "context": context,
            "seed": seed,
            "candidate_score": candidate_score,
            "incumbent_score": incumbent_score,
            "adapters_count": len(self.adapters)
        }
        
        job_id = self._log_job("promote", "running", input_data)
        
        try:
            # Make promotion decision
            promoted = candidate_score > incumbent_score
            
            if promoted:
                # Add new adapter to ring buffer
                new_adapter = AdapterInfo(
                    model_path=model_path,
                    score=candidate_score,
                    timestamp=datetime.utcnow()
                )
                self.adapters.append(new_adapter)
                self._save_adapters()
                
                # Log successful promotion
                output_data = {
                    "promoted": True,
                    "candidate_score": candidate_score,
                    "incumbent_score": incumbent_score,
                    "new_adapters_count": len(self.adapters),
                    "jury_result": {
                        "candidate_score": jury_result.candidate_score,
                        "judge_results": [
                            {
                                "name": r.judge_name,
                                "score": r.score,
                                "reasoning": r.reasoning
                            }
                            for r in jury_result.judge_results
                        ],
                        "aggregation_method": jury_result.aggregation_method,
                        "seed": jury_result.seed
                    }
                }
                
                self._log_job("promote", "completed", input_data, output_data)
                
            else:
                # Log rejection
                output_data = {
                    "promoted": False,
                    "candidate_score": candidate_score,
                    "incumbent_score": incumbent_score,
                    "reason": "candidate_score_not_higher",
                    "jury_result": {
                        "candidate_score": jury_result.candidate_score,
                        "judge_results": [
                            {
                                "name": r.judge_name,
                                "score": r.score,
                                "reasoning": r.reasoning
                            }
                            for r in jury_result.judge_results
                        ],
                        "aggregation_method": jury_result.aggregation_method,
                        "seed": jury_result.seed
                    }
                }
                
                self._log_job("promote", "completed", input_data, output_data)
            
            return {
                "promoted": promoted,
                "candidate_score": candidate_score,
                "incumbent_score": incumbent_score,
                "jury_result": jury_result,
                "adapters_count": len(self.adapters)
            }
            
        except Exception as e:
            # Log failed promotion
            error_data = {
                "type": "promotion_error",
                "message": str(e),
                "stack": None
            }
            
            self._log_job("promote", "failed", input_data, error_data=error_data)
            raise PromotionError(f"Promotion failed: {e}") from e
    
    def get_adapter_history(self) -> List[Dict[str, Any]]:
        """Get the current adapter history."""
        return [
            {
                "model_path": adapter.model_path,
                "score": adapter.score,
                "timestamp": adapter.timestamp.isoformat() + "Z"
            }
            for adapter in self.adapters
        ]
    
    def rollback_to_adapter(self, model_path: str) -> bool:
        """Rollback to a specific adapter in history."""
        # Find the adapter in history
        target_adapter = None
        for adapter in self.adapters:
            if adapter.model_path == model_path:
                target_adapter = adapter
                break
        
        if target_adapter is None:
            return False
        
        # Remove all adapters after the target
        while self.adapters and self.adapters[-1].model_path != model_path:
            self.adapters.pop()
        
        self._save_adapters()
        return True

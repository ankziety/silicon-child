"""Policy module for determining next actions in the research loop."""

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ActionType(Enum):
    """Types of actions that can be performed."""

    SEARCH = "search"
    FETCH = "fetch"
    PARSE = "parse"
    ANSWER = "answer"
    STOP = "stop"


class ActionState(BaseModel):
    """State of an action in the research loop."""

    action_type: ActionType
    input_data: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResearchState(BaseModel):
    """Current state of the research process."""

    question: str
    search_queries: List[str]
    urls_fetched: List[str]
    documents_parsed: List[str]
    quotes_collected: List[Dict[str, Any]]
    answer_generated: Optional[str] = None
    max_iterations: int = 10
    current_iteration: int = 0
    min_quotes_required: int = 2


class Policy:
    """Policy for determining the next action in the research loop."""

    def __init__(self, store: Any):
        """Initialize policy with storage."""
        self.store = store
        self.max_search_queries = 5
        self.max_urls_to_fetch = 10
        self.max_documents_to_parse = 10

    def _log_job(
        self,
        job_type: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        error_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a job to the store."""
        job_id = f"{job_type}-{int(time.time() * 1000)}"

        job_data = {
            "id": job_id,
            "type": job_type,
            "status": "failed" if error_data else "completed",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "error": error_data,
            "metadata": {
                "version": "0.1.0",
                "priority": 5,
                "retries": 0,
                "max_retries": 3,
                "timeout_seconds": 30,
            },
        }

        self.store.store_job(job_data)
        return job_id

    def next_action(self, state: ResearchState) -> Optional[ActionState]:
        """Determine the next action based on current research state."""
        start_time = datetime.utcnow()

        try:
            # Check if we have enough quotes to generate an answer
            if len(state.quotes_collected) >= state.min_quotes_required and not state.answer_generated:
                action = ActionState(
                    action_type=ActionType.ANSWER,
                    input_data={
                        "question": state.question,
                        "quotes": state.quotes_collected,
                        "documents_count": len(state.documents_parsed),
                    },
                    created_at=start_time,
                )
                
                output_data = {
                    "action_type": "answer",
                    "quotes_count": len(state.quotes_collected),
                    "documents_count": len(state.documents_parsed),
                }
                self._log_job("plan", {"state": state.model_dump()}, output_data)
                return action

            # Check if we need to parse more documents
            if len(state.urls_fetched) > len(state.documents_parsed):
                next_url = state.urls_fetched[len(state.documents_parsed)]
                action = ActionState(
                    action_type=ActionType.PARSE,
                    input_data={"url": next_url},
                    created_at=start_time,
                )
                
                output_data = {
                    "action_type": "parse",
                    "url": next_url,
                    "documents_parsed": len(state.documents_parsed),
                    "urls_fetched": len(state.urls_fetched),
                }
                self._log_job("plan", {"state": state.model_dump()}, output_data)
                return action

            # Check if we need to fetch more URLs
            if len(state.urls_fetched) < self.max_urls_to_fetch and len(state.search_queries) > 0:
                # Use the next search query to find more URLs
                next_query = state.search_queries[len(state.urls_fetched) % len(state.search_queries)]
                action = ActionState(
                    action_type=ActionType.FETCH,
                    input_data={"search_query": next_query, "max_results": 3},
                    created_at=start_time,
                )
                
                output_data = {
                    "action_type": "fetch",
                    "search_query": next_query,
                    "urls_fetched": len(state.urls_fetched),
                    "max_urls": self.max_urls_to_fetch,
                }
                self._log_job("plan", {"state": state.model_dump()}, output_data)
                return action

            # Check if we need to generate more search queries
            if len(state.search_queries) < self.max_search_queries:
                action = ActionState(
                    action_type=ActionType.SEARCH,
                    input_data={
                        "question": state.question,
                        "existing_queries": state.search_queries,
                        "max_queries": self.max_search_queries,
                    },
                    created_at=start_time,
                )
                
                output_data = {
                    "action_type": "search",
                    "question": state.question,
                    "existing_queries_count": len(state.search_queries),
                    "max_queries": self.max_search_queries,
                }
                self._log_job("plan", {"state": state.model_dump()}, output_data)
                return action

            # Check if we've reached maximum iterations
            if state.current_iteration >= state.max_iterations:
                action = ActionState(
                    action_type=ActionType.STOP,
                    input_data={
                        "reason": "max_iterations_reached",
                        "iterations": state.current_iteration,
                        "max_iterations": state.max_iterations,
                    },
                    created_at=start_time,
                )
                
                output_data = {
                    "action_type": "stop",
                    "reason": "max_iterations_reached",
                    "iterations": state.current_iteration,
                    "quotes_collected": len(state.quotes_collected),
                }
                self._log_job("plan", {"state": state.model_dump()}, output_data)
                return action

            # If we have some quotes but not enough, try to get more
            if len(state.quotes_collected) > 0 and len(state.quotes_collected) < state.min_quotes_required:
                # Try to fetch more URLs with different search queries
                if len(state.search_queries) > 0:
                    next_query = state.search_queries[len(state.urls_fetched) % len(state.search_queries)]
                    action = ActionState(
                        action_type=ActionType.FETCH,
                        input_data={"search_query": next_query, "max_results": 5},
                        created_at=start_time,
                    )
                    
                    output_data = {
                        "action_type": "fetch",
                        "search_query": next_query,
                        "quotes_needed": state.min_quotes_required - len(state.quotes_collected),
                        "quotes_collected": len(state.quotes_collected),
                    }
                    self._log_job("plan", {"state": state.model_dump()}, output_data)
                    return action

            # Default: stop if we can't make progress
            action = ActionState(
                action_type=ActionType.STOP,
                input_data={
                    "reason": "no_progress_possible",
                    "quotes_collected": len(state.quotes_collected),
                    "documents_parsed": len(state.documents_parsed),
                },
                created_at=start_time,
            )
            
            output_data = {
                "action_type": "stop",
                "reason": "no_progress_possible",
                "quotes_collected": len(state.quotes_collected),
                "documents_parsed": len(state.documents_parsed),
            }
            self._log_job("plan", {"state": state.model_dump()}, output_data)
            return action

        except Exception as e:
            error_data = {"type": "policy_error", "message": str(e), "stack": None}
            self._log_job("plan", {"state": state.model_dump()}, error_data=error_data)
            return None

    def should_continue(self, state: ResearchState) -> bool:
        """Determine if the research loop should continue."""
        # Stop if we have an answer
        if state.answer_generated:
            return False

        # Stop if we've reached max iterations
        if state.current_iteration >= state.max_iterations:
            return False

        # Stop if we have enough quotes and no more actions to take
        if len(state.quotes_collected) >= state.min_quotes_required and not self._has_more_actions(state):
            return False
        
        # Stop if we have enough quotes and no URLs to parse
        if len(state.quotes_collected) >= state.min_quotes_required and len(state.urls_fetched) == len(state.documents_parsed):
            return False

        # Continue if we can still make progress
        return True

    def _has_more_actions(self, state: ResearchState) -> bool:
        """Check if there are more actions that can be taken."""
        # Can still search for more queries
        if len(state.search_queries) < self.max_search_queries:
            return True
        
        # Can still fetch more URLs
        if len(state.urls_fetched) < self.max_urls_to_fetch and len(state.search_queries) > 0:
            return True
        
        # Can still parse more documents
        if len(state.urls_fetched) > len(state.documents_parsed):
            return True
        
        return False

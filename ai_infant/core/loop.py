"""Core loop for orchestrating the research process."""

import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from ..crawl.browser import Browser
from ..plan.policy import ActionState, ActionType, Policy, ResearchState
from ..text.image_analysis import ImageAnalyzer
from ..text.parse import Parser


class TraceEntry(BaseModel):
    """A single trace entry for the research loop."""

    id: str
    job_id: str
    component: str
    operation: str
    status: str
    timestamp: str
    duration_ms: int
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class Answer(BaseModel):
    """Answer with anchored quotes."""

    question: str
    answer: str
    quotes: list[dict[str, Any]]
    documents_used: list[str]
    generated_at: datetime
    trace_id: str


class ResearchLoop:
    """Core research loop that orchestrates the entire process."""

    def __init__(self, store: Any, headless: bool = False):
        """Initialize research loop with storage."""
        self.store = store
        self.browser = Browser(store, headless=headless)
        self.parser = Parser(store)
        self.policy = Policy(store)
        self.image_analyzer = ImageAnalyzer(store)
        self.traces: list[TraceEntry] = []

    def close(self):
        """Close browser and cleanup resources."""
        if hasattr(self, "browser"):
            self.browser.close()

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return f"trace-{uuid.uuid4().hex[:8]}"

    def _log_trace(
        self,
        job_id: str,
        component: str,
        operation: str,
        status: str,
        start_time: datetime,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        error_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a trace entry."""
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        trace = TraceEntry(
            id=self._generate_trace_id(),
            job_id=job_id,
            component=component,
            operation=operation,
            status=status,
            timestamp=end_time.isoformat() + "Z",
            duration_ms=duration_ms,
            input=input_data,
            output=output_data,
            error=error_data,
            metadata={
                "version": "0.1.0",
                "environment": "production",
                "correlation_id": job_id,
            },
        )

        self.traces.append(trace)
        self.store.store_trace(trace.model_dump())

    def _execute_search(self, action: ActionState) -> Optional[dict[str, Any]]:
        """Execute search action to generate search queries."""
        start_time = datetime.utcnow()
        job_id = f"search-{int(time.time() * 1000)}"

        try:
            self._log_trace(
                job_id, "core", "search_start", "started", start_time, action.input_data
            )

            # Simple search query generation based on the question
            question = action.input_data["question"]
            existing_queries = action.input_data.get("existing_queries", [])
            max_queries = action.input_data.get("max_queries", 5)

            # Generate search queries by extracting key terms
            words = question.lower().split()
            # Remove common stop words
            stop_words = {
                "what",
                "is",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            }
            key_words = [
                word for word in words if word not in stop_words and len(word) > 2
            ]

            new_queries = []
            if len(key_words) >= 2:
                # Create queries with 2-3 key words
                for i in range(0, len(key_words), 2):
                    if len(new_queries) >= max_queries - len(existing_queries):
                        break
                    query = " ".join(key_words[i : i + 2])
                    if query not in existing_queries:
                        new_queries.append(query)

            # Add the original question as a query if we have room
            if len(new_queries) < max_queries - len(existing_queries):
                new_queries.append(question)

            result = {
                "queries": new_queries,
                "total_queries": len(existing_queries) + len(new_queries),
            }

            self._log_trace(
                job_id,
                "core",
                "search_complete",
                "completed",
                start_time,
                action.input_data,
                result,
            )
            return result

        except Exception as e:
            error_data = {"type": "search_error", "message": str(e), "stack": None}
            self._log_trace(
                job_id,
                "core",
                "search_failed",
                "failed",
                start_time,
                action.input_data,
                error_data=error_data,
            )
            return None

    def _execute_fetch(self, action: ActionState) -> Optional[dict[str, Any]]:
        """Execute fetch action to retrieve URLs."""
        start_time = datetime.utcnow()
        job_id = f"fetch-{int(time.time() * 1000)}"

        try:
            self._log_trace(
                job_id, "core", "fetch_start", "started", start_time, action.input_data
            )

            search_query = action.input_data["search_query"]
            max_results = action.input_data.get("max_results", 3)

            # Use the browser's search functionality to find real URLs
            urls = []
            try:
                # Use the browser to search for the query
                search_results = self.browser.search(
                    search_query, max_results=max_results
                )
                if search_results:
                    urls = search_results
                else:
                    # Fallback to some reliable sources if search fails
                    urls = [
                        "https://httpbin.org/html",  # Test page
                        "https://httpbin.org/json",  # Test API
                    ]
            except Exception:
                # If search fails, use fallback URLs
                urls = [
                    "https://httpbin.org/html",  # Test page
                    "https://httpbin.org/json",  # Test API
                ]

            # Limit to max_results
            urls = urls[:max_results]

            result = {
                "urls": urls,
                "search_query": search_query,
                "urls_count": len(urls),
            }

            self._log_trace(
                job_id,
                "core",
                "fetch_complete",
                "completed",
                start_time,
                action.input_data,
                result,
            )
            return result

        except Exception as e:
            error_data = {"type": "fetch_error", "message": str(e), "stack": None}
            self._log_trace(
                job_id,
                "core",
                "fetch_failed",
                "failed",
                start_time,
                action.input_data,
                error_data=error_data,
            )
            return None

    def _execute_parse(self, action: ActionState) -> Optional[dict[str, Any]]:
        """Execute parse action to extract content and quotes."""
        start_time = datetime.utcnow()
        job_id = f"parse-{int(time.time() * 1000)}"

        try:
            self._log_trace(
                job_id, "core", "parse_start", "started", start_time, action.input_data
            )

            url = action.input_data["url"]

            # Fetch the content first
            fetch_result = self.browser.fetch(url)
            if not fetch_result:
                raise Exception(f"Failed to fetch URL: {url}")

            # Parse the content
            parsed_doc = self.parser.parse(
                url, fetch_result.content, fetch_result.mime_type
            )
            if not parsed_doc:
                raise Exception(f"Failed to parse content from URL: {url}")

            # Analyze screenshot if available
            image_analysis = None
            if fetch_result.screenshot_path:
                image_analysis = self.image_analyzer.analyze_image(
                    fetch_result.screenshot_path
                )

            # Store the document
            doc_data = {
                "id": f"doc-{hashlib.sha256(url.encode()).hexdigest()[:8]}",
                "url": url,
                "content": parsed_doc.content,
                "metadata": {
                    "source": "browser",
                    "mime_type": fetch_result.mime_type,
                    "size_bytes": fetch_result.size_bytes,
                    "title": parsed_doc.title,
                    "author": parsed_doc.author,
                    "language": parsed_doc.language,
                    "checksum": parsed_doc.checksum,
                    "screenshot_path": fetch_result.screenshot_path,
                    "image_analysis": (
                        image_analysis.model_dump() if image_analysis else None
                    ),
                },
                "timestamp": parsed_doc.parse_time.isoformat() + "Z",
                "processing": {
                    "version": "0.1.0",
                    "stage": "parsed",
                },
            }

            self.store.store_document(doc_data)

            result = {
                "url": url,
                "title": parsed_doc.title,
                "quotes": parsed_doc.quotes,
                "quotes_count": len(parsed_doc.quotes),
                "content_length": len(parsed_doc.content),
                "document_id": doc_data["id"],
                "screenshot_path": fetch_result.screenshot_path,
                "image_analysis": (
                    image_analysis.model_dump() if image_analysis else None
                ),
            }

            self._log_trace(
                job_id,
                "core",
                "parse_complete",
                "completed",
                start_time,
                action.input_data,
                result,
            )
            return result

        except Exception as e:
            error_data = {"type": "parse_error", "message": str(e), "stack": None}
            self._log_trace(
                job_id,
                "core",
                "parse_failed",
                "failed",
                start_time,
                action.input_data,
                error_data=error_data,
            )
            return None

    def _execute_answer(self, action: ActionState) -> Optional[dict[str, Any]]:
        """Execute answer action to generate final answer with quotes."""
        start_time = datetime.utcnow()
        job_id = f"answer-{int(time.time() * 1000)}"

        try:
            self._log_trace(
                job_id, "core", "answer_start", "started", start_time, action.input_data
            )

            question = action.input_data["question"]
            quotes = action.input_data["quotes"]
            documents_count = action.input_data["documents_count"]

            # Generate answer based on quotes
            if not quotes:
                answer_text = (
                    "I couldn't find enough information to answer this question."
                )
            else:
                # Simple answer generation - in a real implementation, this would use an LLM
                quote_texts = [
                    quote["text"] for quote in quotes[:3]
                ]  # Use top 3 quotes
                answer_text = f"Based on the research, I found the following information: {' '.join(quote_texts)}"

            # Create answer object
            answer = Answer(
                question=question,
                answer=answer_text,
                quotes=quotes,
                documents_used=[f"doc-{i}" for i in range(documents_count)],
                generated_at=datetime.utcnow(),
                trace_id=job_id,
            )

            result = {
                "answer": answer.answer,
                "quotes_count": len(answer.quotes),
                "documents_used": answer.documents_used,
                "trace_id": answer.trace_id,
            }

            self._log_trace(
                job_id,
                "core",
                "answer_complete",
                "completed",
                start_time,
                action.input_data,
                result,
            )

            # Log the answer job
            job_data = {
                "id": job_id,
                "type": "answer",
                "status": "completed",
                "created_at": start_time.isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "started_at": start_time.isoformat() + "Z",
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "input": action.input_data,
                "output": result,
                "error": None,
                "metadata": {
                    "version": "0.1.0",
                    "priority": 5,
                    "retries": 0,
                    "max_retries": 3,
                    "timeout_seconds": 30,
                },
            }
            self.store.store_job(job_data)

            return result

        except Exception as e:
            error_data = {"type": "answer_error", "message": str(e), "stack": None}
            self._log_trace(
                job_id,
                "core",
                "answer_failed",
                "failed",
                start_time,
                action.input_data,
                error_data=error_data,
            )
            return None

    def _execute_action(self, action: ActionState) -> Optional[dict[str, Any]]:
        """Execute a single action based on its type."""
        if action.action_type == ActionType.SEARCH:
            return self._execute_search(action)
        elif action.action_type == ActionType.FETCH:
            return self._execute_fetch(action)
        elif action.action_type == ActionType.PARSE:
            return self._execute_parse(action)
        elif action.action_type == ActionType.ANSWER:
            return self._execute_answer(action)
        elif action.action_type == ActionType.STOP:
            return {"reason": action.input_data.get("reason", "unknown")}
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    def research(
        self, question: str, max_iterations: int = 10, min_quotes: int = 2
    ) -> Optional[Answer]:
        """Execute the complete research loop for a question."""
        start_time = datetime.utcnow()
        job_id = f"research-{int(time.time() * 1000)}"

        try:
            self._log_trace(
                job_id,
                "core",
                "research_start",
                "started",
                start_time,
                {"question": question},
            )

            # Initialize research state
            state = ResearchState(
                question=question,
                search_queries=[],
                urls_fetched=[],
                documents_parsed=[],
                quotes_collected=[],
                max_iterations=max_iterations,
                min_quotes_required=min_quotes,
            )

            answer_result = None

            # Main research loop
            while self.policy.should_continue(state):
                # Get next action from policy
                action = self.policy.next_action(state)
                if not action:
                    break

                # Execute the action
                result = self._execute_action(action)
                if not result:
                    break

                # Update state based on action result
                if action.action_type == ActionType.SEARCH:
                    state.search_queries.extend(result.get("queries", []))
                elif action.action_type == ActionType.FETCH:
                    state.urls_fetched.extend(result.get("urls", []))
                elif action.action_type == ActionType.PARSE:
                    state.documents_parsed.append(result.get("url", ""))
                    state.quotes_collected.extend(result.get("quotes", []))
                elif action.action_type == ActionType.ANSWER:
                    answer_result = result
                    state.answer_generated = result.get("answer", "")

                state.current_iteration += 1

            # Create final answer if we have one
            if answer_result:
                answer = Answer(
                    question=question,
                    answer=answer_result["answer"],
                    quotes=state.quotes_collected,
                    documents_used=answer_result["documents_used"],
                    generated_at=datetime.utcnow(),
                    trace_id=job_id,
                )
            else:
                # Create a fallback answer if we couldn't generate one
                answer = Answer(
                    question=question,
                    answer="I couldn't find enough information to answer this question.",
                    quotes=state.quotes_collected,
                    documents_used=[],
                    generated_at=datetime.utcnow(),
                    trace_id=job_id,
                )

            result_data = {
                "answer": answer.answer,
                "quotes_count": len(answer.quotes),
                "documents_used": answer.documents_used,
                "iterations": state.current_iteration,
                "traces_count": len(self.traces),
            }

            self._log_trace(
                job_id,
                "core",
                "research_complete",
                "completed",
                start_time,
                {"question": question},
                result_data,
            )
            return answer

        except Exception as e:
            error_data = {"type": "research_error", "message": str(e), "stack": None}
            self._log_trace(
                job_id,
                "core",
                "research_failed",
                "failed",
                start_time,
                {"question": question},
                error_data=error_data,
            )
            return None

    def research_autonomously(self, question: str) -> Optional[Answer]:
        """Execute autonomous research - the system decides how much to research."""
        start_time = datetime.utcnow()
        job_id = f"autonomous-research-{int(time.time() * 1000)}"

        try:
            # Temporarily disable duplicate detection to debug
            # Check if we've already researched this question recently
            # recent_jobs = self.store.get_recent_jobs(limit=20)
            # recent_questions = []
            # for job in recent_jobs:
            #     if job['type'] == 'research' and 'input' in job and 'question' in job['input']:
            #         recent_questions.append(job['input']['question'].lower())
            #
            # # If we've researched this question recently, skip it
            # if question.lower() in recent_questions:
            #     print(f"Skipping duplicate research: {question}")
            #     return None

            self._log_trace(
                job_id,
                "core",
                "autonomous_research_start",
                "started",
                start_time,
                {"question": question},
            )

            # Initialize research state with no artificial limits
            state = ResearchState(
                question=question,
                search_queries=[],
                urls_fetched=[],
                documents_parsed=[],
                quotes_collected=[],
                max_iterations=100,  # High limit, but system decides when to stop
                min_quotes_required=0,  # No minimum requirement
            )

            answer_result = None
            consecutive_failures = 0

            # Autonomous research loop - system decides when to stop
            while self.policy.should_continue_autonomously(state, consecutive_failures):
                # Get next action from policy
                action = self.policy.next_action(state)
                if not action:
                    break

                # Execute the action
                result = self._execute_action(action)
                if not result:
                    consecutive_failures += 1
                    continue

                consecutive_failures = 0  # Reset on success

                # Update state based on action result
                if action.action_type == ActionType.SEARCH:
                    state.search_queries.extend(result.get("queries", []))
                elif action.action_type == ActionType.FETCH:
                    # Filter out URLs we've already fetched to avoid repetition
                    new_urls = []
                    for url in result.get("urls", []):
                        if url not in state.urls_fetched:
                            new_urls.append(url)
                    state.urls_fetched.extend(new_urls)
                elif action.action_type == ActionType.PARSE:
                    state.documents_parsed.append(result.get("url", ""))
                    state.quotes_collected.extend(result.get("quotes", []))
                elif action.action_type == ActionType.ANSWER:
                    answer_result = result
                    state.answer_generated = result.get("answer", "")

                state.current_iteration += 1

            # Create final answer if we have one
            if answer_result:
                answer = Answer(
                    question=question,
                    answer=answer_result["answer"],
                    quotes=state.quotes_collected,
                    documents_used=answer_result["documents_used"],
                    generated_at=datetime.utcnow(),
                    trace_id=job_id,
                )
            else:
                # Create a fallback answer if we couldn't generate one
                answer = Answer(
                    question=question,
                    answer="I explored this topic but couldn't find enough information to provide a comprehensive answer.",
                    quotes=state.quotes_collected,
                    documents_used=[],
                    generated_at=datetime.utcnow(),
                    trace_id=job_id,
                )

            result_data = {
                "answer": answer.answer,
                "quotes_count": len(answer.quotes),
                "documents_used": answer.documents_used,
                "iterations": state.current_iteration,
                "traces_count": len(self.traces),
                "autonomous": True,
            }

            self._log_trace(
                job_id,
                "core",
                "autonomous_research_complete",
                "completed",
                start_time,
                {"question": question},
                result_data,
            )
            return answer

        except Exception as e:
            error_data = {
                "type": "autonomous_research_error",
                "message": str(e),
                "stack": None,
            }
            self._log_trace(
                job_id,
                "core",
                "autonomous_research_failed",
                "failed",
                start_time,
                {"question": question},
                error_data=error_data,
            )
            return None

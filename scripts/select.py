#!/usr/bin/env python3
"""Dataset selector for LoRA training - filters top-scoring traces from store."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_infant.data import Store


class DatasetSelector:
    """Selects and filters traces for training dataset creation."""

    def __init__(self, store: Store):
        """Initialize dataset selector with store."""
        self.store = store

    def get_top_scoring_traces(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Get top-scoring traces from store with fallback to recent traces."""
        # First try to get traces with evaluation scores
        traces_with_scores = self._get_traces_with_scores(limit)

        if traces_with_scores:
            return traces_with_scores

        # Fallback: get most recent traces
        print("No traces with evaluation scores found, using most recent traces")
        return self._get_recent_traces(limit)

    def _get_traces_with_scores(self, limit: int) -> list[dict[str, Any]]:
        """Get traces that have evaluation scores."""
        # Query traces table for entries with evaluation metadata
        query = """
        SELECT * FROM traces
        WHERE metadata IS NOT NULL
        AND metadata LIKE '%score%'
        ORDER BY CAST(JSON_EXTRACT(metadata, '$.score') AS FLOAT) DESC
        LIMIT ?
        """

        try:
            result = self.store.conn.execute(query, [limit]).fetchall()
            return [
                self.store._row_to_dict(row, self.store.conn.description)
                for row in result
            ]
        except Exception as e:
            print(f"Error querying traces with scores: {e}")
            return []

    def _get_recent_traces(self, limit: int) -> list[dict[str, Any]]:
        """Get most recent traces as fallback."""
        query = """
        SELECT * FROM traces
        ORDER BY timestamp DESC
        LIMIT ?
        """

        try:
            result = self.store.conn.execute(query, [limit]).fetchall()
            return [
                self.store._row_to_dict(row, self.store.conn.description)
                for row in result
            ]
        except Exception as e:
            print(f"Error querying recent traces: {e}")
            return []

    def convert_traces_to_jsonl(
        self, traces: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Convert traces to training format with input/output pairs."""
        training_data = []

        for trace in traces:
            # Extract input and output from trace
            input_data = trace.get("input", {})
            output_data = trace.get("output", {})

            # Convert to string format for training
            input_text = self._extract_text_from_data(input_data)
            output_text = self._extract_text_from_data(output_data)

            if input_text and output_text:
                training_data.append({"input": input_text, "output": output_text})

        return training_data

    def _extract_text_from_data(self, data: Any) -> str:
        """Extract text content from trace data."""
        if isinstance(data, dict):
            # Look for common text fields
            for field in ["text", "content", "response", "answer", "query"]:
                if field in data and data[field]:
                    return str(data[field])

            # If no specific field, try to convert entire dict to string
            return json.dumps(data, ensure_ascii=False)

        elif isinstance(data, str):
            return data

        elif data is not None:
            return str(data)

        return ""

    def save_jsonl(self, data: list[dict[str, str]], output_path: Path) -> None:
        """Save training data to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def log_job(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> str:
        """Log dataset selection job."""
        job_id = f"select-{int(time.time() * 1000)}"
        now = datetime.utcnow().isoformat() + "Z"

        job_data = {
            "id": job_id,
            "type": "select",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
            "input": input_data,
            "output": output_data,
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
        return job_id


def main():
    """Main entry point for dataset selection."""
    parser = argparse.ArgumentParser(
        description="Select traces for LoRA training dataset"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/training_dataset.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=1000,
        help="Maximum number of traces to select",
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default="data/ai_infant.db",
        help="Path to store database",
    )

    args = parser.parse_args()

    # Initialize store and selector
    store = Store(args.store_path)
    selector = DatasetSelector(store)

    print(f"Selecting up to {args.limit} traces for training dataset...")

    # Get traces
    traces = selector.get_top_scoring_traces(args.limit)

    if not traces:
        print("No traces found in store")
        # Create empty dataset and log job
        selector.save_jsonl([], args.output)
        selector.log_job(
            {"limit": args.limit, "store_path": args.store_path},
            {
                "trace_count": 0,
                "output_path": str(args.output),
                "warning": "No traces found",
            },
        )
        return 0

    # Convert to training format
    training_data = selector.convert_traces_to_jsonl(traces)

    if not training_data:
        print("No valid training examples found in traces")
        selector.save_jsonl([], args.output)
        selector.log_job(
            {"limit": args.limit, "store_path": args.store_path},
            {
                "trace_count": len(traces),
                "training_examples": 0,
                "output_path": str(args.output),
                "warning": "No valid training examples",
            },
        )
        return 0

    # Save dataset
    selector.save_jsonl(training_data, args.output)

    # Log job
    selector.log_job(
        {"limit": args.limit, "store_path": args.store_path},
        {
            "trace_count": len(traces),
            "training_examples": len(training_data),
            "output_path": str(args.output),
            "dataset_size_mb": args.output.stat().st_size / (1024 * 1024)
            if args.output.exists()
            else 0,
        },
    )

    print(f"Created training dataset with {len(training_data)} examples")
    print(f"Dataset saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

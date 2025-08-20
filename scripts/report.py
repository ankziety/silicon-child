"""Weekly report generator for AI-Infant research agent."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ai_infant.data.store import Store


class ReportGenerator:
    """Generates weekly reports with key metrics."""

    def __init__(self, store: Store, reports_dir: str = "reports"):
        """Initialize report generator."""
        self.store = store
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _log_job(
        self,
        job_type: str,
        status: str,
        input_data: dict[str, Any],
        output_data: Optional[dict[str, Any]] = None,
        error_data: Optional[dict[str, Any]] = None,
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
            "metadata": {"version": "0.1.0", "priority": 5},
        }

        self.store.store_job(job_data)
        return job_id

    def _get_weekly_date_range(self) -> tuple[datetime, datetime]:
        """Get the date range for the current week (Monday to Sunday)."""
        now = datetime.utcnow()
        # Find the most recent Monday
        days_since_monday = now.weekday()
        week_start = now - timedelta(days=days_since_monday)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        return week_start, week_end

    def _calculate_tokens_per_day(
        self, week_start: datetime, week_end: datetime
    ) -> dict[str, float]:
        """Calculate tokens per day from document content."""
        documents = self.store.get_documents()

        # Filter documents within the week
        weekly_docs = []
        for doc in documents:
            try:
                doc_timestamp = datetime.fromisoformat(
                    doc["timestamp"].replace("Z", "+00:00")
                )
                if week_start <= doc_timestamp < week_end:
                    weekly_docs.append(doc)
            except (ValueError, TypeError):
                # Skip documents with invalid timestamps
                continue

        # Calculate total tokens (rough estimate: 1 token ≈ 4 characters)
        total_tokens = sum(len(doc["content"]) // 4 for doc in weekly_docs)
        days_in_week = 7

        return {
            "tokens_per_day": total_tokens / days_in_week if days_in_week > 0 else 0.0,
            "total_tokens": total_tokens,
            "documents_processed": len(weekly_docs),
        }

    def _calculate_pages_per_day(
        self, week_start: datetime, week_end: datetime
    ) -> dict[str, float]:
        """Calculate pages per day from fetch jobs."""
        fetch_jobs = self.store.get_jobs(job_type="fetch")

        # Filter jobs within the week
        weekly_fetches = []
        for job in fetch_jobs:
            try:
                job_timestamp = datetime.fromisoformat(
                    job["created_at"].replace("Z", "+00:00")
                )
                if (
                    week_start <= job_timestamp < week_end
                    and job["status"] == "completed"
                ):
                    weekly_fetches.append(job)
            except (ValueError, TypeError):
                # Skip jobs with invalid timestamps
                continue

        days_in_week = 7

        return {
            "pages_per_day": (
                len(weekly_fetches) / days_in_week if days_in_week > 0 else 0.0
            ),
            "total_pages": len(weekly_fetches),
            "successful_fetches": len(weekly_fetches),
        }

    def _get_eval_score_delta(self) -> dict[str, Any]:
        """Get evaluation score delta between last and current adapter."""
        # Get all eval jobs
        eval_jobs = self.store.get_jobs(job_type="eval")

        if not eval_jobs:
            return {
                "current_score": 0.0,
                "previous_score": 0.0,
                "delta": 0.0,
                "current_adapter_id": None,
                "previous_adapter_id": None,
            }

        # Sort by creation time (newest first)
        eval_jobs.sort(key=lambda x: x["created_at"], reverse=True)

        current_score = 0.0
        previous_score = 0.0
        current_adapter_id = None
        previous_adapter_id = None

        # Find the most recent successful eval
        for job in eval_jobs:
            if job["status"] == "completed" and job["output"]:
                current_score = job["output"].get("candidate_score", 0.0)
                current_adapter_id = job["input"].get("model_path", "unknown")
                break

        # Find the second most recent successful eval
        for job in eval_jobs[1:]:
            if job["status"] == "completed" and job["output"]:
                previous_score = job["output"].get("candidate_score", 0.0)
                previous_adapter_id = job["input"].get("model_path", "unknown")
                break

        delta = current_score - previous_score

        return {
            "current_score": current_score,
            "previous_score": previous_score,
            "delta": delta,
            "current_adapter_id": current_adapter_id,
            "previous_adapter_id": previous_adapter_id,
        }

    def _get_current_adapter_id(self) -> str:
        """Get the current adapter ID from adapters.json."""
        adapters_file = Path("data/adapters.json")
        if not adapters_file.exists():
            return "none"

        try:
            with open(adapters_file) as f:
                data = json.load(f)
                adapters = data.get("adapters", [])
                if adapters:
                    # Return the most recent adapter
                    return adapters[-1].get("model_path", "unknown")
                return "none"
        except (json.JSONDecodeError, KeyError):
            return "error"

    def _get_rollback_history(self) -> list[dict[str, Any]]:
        """Get rollback history from promotion jobs."""
        promote_jobs = self.store.get_jobs(job_type="promote")

        rollbacks = []
        for job in promote_jobs:
            if job["status"] == "completed" and job["output"]:
                output = job["output"]
                if output.get("promoted", False) and output.get(
                    "rollback_triggered", False
                ):
                    rollbacks.append(
                        {
                            "timestamp": job["created_at"],
                            "model_path": job["input"].get("model_path", "unknown"),
                            "score": output.get("candidate_score", 0.0),
                            "reason": output.get("rollback_reason", "unknown"),
                        }
                    )

        return rollbacks

    def _calculate_disk_usage(self) -> dict[str, Any]:
        """Calculate disk usage for data directory."""
        data_dir = Path("data")
        if not data_dir.exists():
            return {"total_bytes": 0, "files": []}

        total_bytes = 0
        files = []

        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_bytes += file_size
                files.append(
                    {
                        "path": str(file_path.relative_to(data_dir)),
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024),
                    }
                )

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "files": sorted(files, key=lambda x: x["size_bytes"], reverse=True),
        }

    def generate_report(self) -> str:
        """Generate a comprehensive weekly report."""
        # Log report generation start
        input_data = {
            "reports_dir": str(self.reports_dir),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self._log_job("report", "running", input_data)

        try:
            # Get date range for the week
            week_start, week_end = self._get_weekly_date_range()

            # Calculate metrics
            tokens_metrics = self._calculate_tokens_per_day(week_start, week_end)
            pages_metrics = self._calculate_pages_per_day(week_start, week_end)
            eval_metrics = self._get_eval_score_delta()
            current_adapter = self._get_current_adapter_id()
            rollback_history = self._get_rollback_history()
            disk_usage = self._calculate_disk_usage()

            # Generate report content
            report_date = week_start.strftime("%Y-%m-%d")
            report_filename = f"{report_date}.md"
            report_path = self.reports_dir / report_filename

            report_content = self._format_report(
                report_date,
                week_start,
                week_end,
                tokens_metrics,
                pages_metrics,
                eval_metrics,
                current_adapter,
                rollback_history,
                disk_usage,
            )

            # Write report to file
            with open(report_path, "w") as f:
                f.write(report_content)

            # Log successful completion
            output_data = {
                "report_path": str(report_path),
                "report_date": report_date,
                "week_start": week_start.isoformat() + "Z",
                "week_end": week_end.isoformat() + "Z",
                "metrics": {
                    "tokens_per_day": tokens_metrics["tokens_per_day"],
                    "pages_per_day": pages_metrics["pages_per_day"],
                    "eval_delta": eval_metrics["delta"],
                    "current_adapter": current_adapter,
                    "rollback_count": len(rollback_history),
                    "disk_usage_mb": disk_usage["total_mb"],
                },
            }

            self._log_job("report", "completed", input_data, output_data)

            return str(report_path)

        except Exception as e:
            # Log failure
            error_data = {
                "type": "report_generation_error",
                "message": str(e),
                "stack": None,
            }

            self._log_job("report", "failed", input_data, error_data=error_data)
            raise

    def _format_report(
        self,
        report_date: str,
        week_start: datetime,
        week_end: datetime,
        tokens_metrics: dict[str, Any],
        pages_metrics: dict[str, Any],
        eval_metrics: dict[str, Any],
        current_adapter: str,
        rollback_history: list[dict[str, Any]],
        disk_usage: dict[str, Any],
    ) -> str:
        """Format the report content as Markdown."""
        return f"""# AI-Infant Weekly Report - {report_date}

**Report Period:** {week_start.strftime("%Y-%m-%d")} to {week_end.strftime("%Y-%m-%d")}
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Key Metrics

### Content Processing
- **Tokens/Day:** {tokens_metrics["tokens_per_day"]:.1f}
- **Total Tokens:** {tokens_metrics["total_tokens"]:,}
- **Documents Processed:** {tokens_metrics["documents_processed"]}

### Web Crawling
- **Pages/Day:** {pages_metrics["pages_per_day"]:.1f}
- **Total Pages:** {pages_metrics["total_pages"]}
- **Successful Fetches:** {pages_metrics["successful_fetches"]}

### Model Evaluation
- **Current Adapter:** `{current_adapter}`
- **Current Score:** {eval_metrics["current_score"]:.3f}
- **Previous Score:** {eval_metrics["previous_score"]:.3f}
- **Score Delta:** {eval_metrics["delta"]:+.3f}

## Rollback History

{self._format_rollback_history(rollback_history)}

## Disk Usage

**Total:** {disk_usage["total_mb"]:.1f} MB

### Largest Files
{self._format_disk_usage(disk_usage)}

## System Status

- **Database Records:** {self._get_record_counts()}
- **Last Activity:** {self._get_last_activity()}

---

*Report generated by AI-Infant research agent*
"""

    def _format_rollback_history(self, rollbacks: list[dict[str, Any]]) -> str:
        """Format rollback history for the report."""
        if not rollbacks:
            return "No rollbacks recorded this week."

        lines = []
        for rollback in rollbacks:
            try:
                timestamp = datetime.fromisoformat(
                    rollback["timestamp"].replace("Z", "+00:00")
                )
                lines.append(
                    f"- **{timestamp.strftime('%Y-%m-%d %H:%M')}**: {rollback['model_path']} (score: {rollback['score']:.3f}) - {rollback['reason']}"
                )
            except (ValueError, TypeError):
                lines.append(
                    f"- **Invalid timestamp**: {rollback['model_path']} (score: {rollback['score']:.3f}) - {rollback['reason']}"
                )

        return "\n".join(lines)

    def _format_disk_usage(self, disk_usage: dict[str, Any]) -> str:
        """Format disk usage for the report."""
        lines = []
        for file_info in disk_usage["files"][:10]:  # Top 10 files
            lines.append(f"- `{file_info['path']}`: {file_info['size_mb']:.1f} MB")

        return "\n".join(lines)

    def _get_record_counts(self) -> str:
        """Get record counts from the database."""
        jobs = self.store.get_jobs()
        traces = self.store.get_traces()
        documents = self.store.get_documents()

        return f"Jobs: {len(jobs)}, Traces: {len(traces)}, Documents: {len(documents)}"

    def _get_last_activity(self) -> str:
        """Get the timestamp of the last activity."""
        jobs = self.store.get_jobs()
        if not jobs:
            return "No activity recorded"

        # Find the most recent job
        latest_job = max(jobs, key=lambda x: x["created_at"])
        try:
            timestamp = datetime.fromisoformat(
                latest_job["created_at"].replace("Z", "+00:00")
            )
            return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, TypeError):
            return "Invalid timestamp format"


def main():
    """Main entry point for report generation."""
    store = Store()

    try:
        generator = ReportGenerator(store)
        report_path = generator.generate_report()
        print(f"Report generated: {report_path}")
    finally:
        store.close()


if __name__ == "__main__":
    main()

"""Retention management for AI-Infant research agent."""

import time
from datetime import datetime
from typing import Any, Optional

from ai_infant.data.store import Store


class RetentionManager:
    """Manages retention and pruning of low-value traces."""

    def __init__(self, store: Store):
        """Initialize retention manager."""
        self.store = store

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

    def _find_duplicate_traces(self) -> list[str]:
        """Find traces that are exact duplicates based on SHA-256 hash."""
        return self.store.get_duplicate_traces()

    def _find_low_scoring_traces(self, percentile: float = 25.0) -> list[str]:
        """Find the bottom percentile of traces by score."""
        traces = self.store.get_traces()

        if not traces:
            return []

        # Calculate scores for all traces
        scored_traces = []
        for trace in traces:
            score = self.store.calculate_trace_score(trace)
            scored_traces.append((trace["id"], score))

        # Sort by score (ascending)
        scored_traces.sort(key=lambda x: x[1])

        # Calculate how many traces to remove
        total_traces = len(scored_traces)
        traces_to_remove = int(total_traces * (percentile / 100.0))

        # Return IDs of the lowest scoring traces
        return [trace_id for trace_id, _ in scored_traces[:traces_to_remove]]

    def run_retention(
        self,
        remove_duplicates: bool = True,
        remove_low_scoring: bool = True,
        low_scoring_percentile: float = 25.0,
    ) -> dict[str, Any]:
        """Run retention process to prune low-value traces."""
        # Log retention start
        input_data = {
            "remove_duplicates": remove_duplicates,
            "remove_low_scoring": remove_low_scoring,
            "low_scoring_percentile": low_scoring_percentile,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self._log_job("retention", "running", input_data)

        try:
            # Get initial disk usage
            initial_disk_usage = self.store.get_disk_usage()
            initial_trace_count = len(self.store.get_traces())

            traces_to_remove = []
            removal_reasons = {}

            # Find duplicate traces
            if remove_duplicates:
                duplicate_ids = self._find_duplicate_traces()
                traces_to_remove.extend(duplicate_ids)
                removal_reasons.update(dict.fromkeys(duplicate_ids, "duplicate"))

            # Find low-scoring traces
            if remove_low_scoring:
                low_scoring_ids = self._find_low_scoring_traces(low_scoring_percentile)
                # Only add if not already marked for removal
                for trace_id in low_scoring_ids:
                    if trace_id not in traces_to_remove:
                        traces_to_remove.append(trace_id)
                        removal_reasons[trace_id] = "low_score"

            # Remove traces
            removed_count = self.store.remove_traces(traces_to_remove)

            # Get final disk usage
            final_disk_usage = self.store.get_disk_usage()
            final_trace_count = len(self.store.get_traces())

            # Calculate disk savings
            disk_savings_mb = (
                initial_disk_usage["total_mb"] - final_disk_usage["total_mb"]
            )

            # Log successful completion
            output_data = {
                "initial_trace_count": initial_trace_count,
                "final_trace_count": final_trace_count,
                "traces_removed": removed_count,
                "duplicates_removed": len(
                    [
                        tid
                        for tid, reason in removal_reasons.items()
                        if reason == "duplicate"
                    ]
                ),
                "low_scoring_removed": len(
                    [
                        tid
                        for tid, reason in removal_reasons.items()
                        if reason == "low_score"
                    ]
                ),
                "initial_disk_usage_mb": initial_disk_usage["total_mb"],
                "final_disk_usage_mb": final_disk_usage["total_mb"],
                "disk_savings_mb": disk_savings_mb,
                "removal_reasons": removal_reasons,
            }

            self._log_job("retention", "completed", input_data, output_data)

            return output_data

        except Exception as e:
            # Log failure
            error_data = {"type": "retention_error", "message": str(e), "stack": None}

            self._log_job("retention", "failed", input_data, error_data=error_data)
            raise

    def get_retention_stats(self) -> dict[str, Any]:
        """Get current retention statistics."""
        traces = self.store.get_traces()

        if not traces:
            return {
                "total_traces": 0,
                "duplicate_count": 0,
                "low_scoring_count": 0,
                "average_score": 0.0,
                "disk_usage_mb": 0.0,
            }

        # Calculate scores
        scores = [self.store.calculate_trace_score(trace) for trace in traces]
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Find duplicates
        duplicate_ids = self._find_duplicate_traces()

        # Find low scoring traces (bottom 25%)
        low_scoring_ids = self._find_low_scoring_traces(25.0)

        # Get disk usage
        disk_usage = self.store.get_disk_usage()

        return {
            "total_traces": len(traces),
            "duplicate_count": len(duplicate_ids),
            "low_scoring_count": len(low_scoring_ids),
            "average_score": average_score,
            "disk_usage_mb": disk_usage["total_mb"],
            "score_distribution": {
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "median": sorted(scores)[len(scores) // 2] if scores else 0.0,
            },
        }

    def create_retention_report(self) -> str:
        """Create a retention analysis report."""
        stats = self.get_retention_stats()

        report = f"""# Retention Analysis Report

**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Current State

- **Total Traces:** {stats["total_traces"]:,}
- **Duplicate Traces:** {stats["duplicate_count"]:,}
- **Low-Scoring Traces (Bottom 25%):** {stats["low_scoring_count"]:,}
- **Average Trace Score:** {stats["average_score"]:.3f}
- **Disk Usage:** {stats["disk_usage_mb"]:.1f} MB

## Score Distribution

- **Minimum Score:** {stats["score_distribution"]["min"]:.3f}
- **Maximum Score:** {stats["score_distribution"]["max"]:.3f}
- **Median Score:** {stats["score_distribution"]["median"]:.3f}

## Potential Savings

If retention is run:
- **Duplicates to Remove:** {stats["duplicate_count"]:,} traces
- **Low-Scoring to Remove:** {stats["low_scoring_count"]:,} traces
- **Total Potential Removal:** {stats["duplicate_count"] + stats["low_scoring_count"]:,} traces
- **Remaining Traces:** {stats["total_traces"] - (stats["duplicate_count"] + stats["low_scoring_count"]):,} traces

## Recommendations

- Run retention to free up storage space
- Consider adjusting scoring algorithm if needed
- Monitor trace generation to reduce duplicates
"""

        return report


def main():
    """Main entry point for retention management."""
    import argparse

    parser = argparse.ArgumentParser(description="AI-Infant Retention Manager")
    parser.add_argument(
        "--stats", action="store_true", help="Show retention statistics"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate retention report"
    )
    parser.add_argument("--run", action="store_true", help="Run retention process")
    parser.add_argument(
        "--no-duplicates", action="store_true", help="Skip duplicate removal"
    )
    parser.add_argument(
        "--no-low-scoring", action="store_true", help="Skip low-scoring removal"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=25.0,
        help="Percentile for low-scoring removal",
    )

    args = parser.parse_args()

    store = Store()

    try:
        manager = RetentionManager(store)

        if args.stats:
            stats = manager.get_retention_stats()
            print("Retention Statistics:")
            print(f"  Total Traces: {stats['total_traces']:,}")
            print(f"  Duplicates: {stats['duplicate_count']:,}")
            print(f"  Low-Scoring: {stats['low_scoring_count']:,}")
            print(f"  Average Score: {stats['average_score']:.3f}")
            print(f"  Disk Usage: {stats['disk_usage_mb']:.1f} MB")

        elif args.report:
            report = manager.create_retention_report()
            print(report)

        elif args.run:
            result = manager.run_retention(
                remove_duplicates=not args.no_duplicates,
                remove_low_scoring=not args.no_low_scoring,
                low_scoring_percentile=args.percentile,
            )
            print("Retention completed successfully!")
            print(f"  Traces removed: {result['traces_removed']:,}")
            print(f"  Duplicates removed: {result['duplicates_removed']:,}")
            print(f"  Low-scoring removed: {result['low_scoring_removed']:,}")
            print(f"  Disk savings: {result['disk_savings_mb']:.1f} MB")

        else:
            # Default: show stats
            stats = manager.get_retention_stats()
            print("Retention Statistics:")
            print(f"  Total Traces: {stats['total_traces']:,}")
            print(f"  Duplicates: {stats['duplicate_count']:,}")
            print(f"  Low-Scoring: {stats['low_scoring_count']:,}")
            print(f"  Average Score: {stats['average_score']:.3f}")
            print(f"  Disk Usage: {stats['disk_usage_mb']:.1f} MB")

    finally:
        store.close()


if __name__ == "__main__":
    main()

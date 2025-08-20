"""Tests for report and retention functionality."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ai_infant.data.store import Store
from scripts.report import ReportGenerator
from scripts.retention import RetentionManager


class TestReportGenerator:
    """Test the report generator functionality."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        # Create a temporary directory for the database
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"

        store = Store(str(db_path))
        yield store
        store.close()

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_data(self, temp_store):
        """Create sample data for testing."""
        # Add sample jobs
        jobs = [
            {
                "id": "fetch-1",
                "type": "fetch",
                "status": "completed",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "input": {"url": "https://example.com"},
                "output": {"status_code": 200, "size_bytes": 1000},
                "metadata": {"version": "0.1.0"},
            },
            {
                "id": "eval-1",
                "type": "eval",
                "status": "completed",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "input": {"model_path": "adapters/test", "prompt": "test"},
                "output": {"candidate_score": 0.85},
                "metadata": {"version": "0.1.0"},
            },
            {
                "id": "eval-2",
                "type": "eval",
                "status": "completed",
                "created_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
                "updated_at": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
                "input": {"model_path": "adapters/old", "prompt": "test"},
                "output": {"candidate_score": 0.75},
                "metadata": {"version": "0.1.0"},
            },
        ]

        for job in jobs:
            temp_store.store_job(job)

        # Add sample documents
        documents = [
            {
                "id": "doc-1",
                "url": "https://example.com/1",
                "content": "This is a test document with some content for token calculation.",
                "metadata": {
                    "source": "test",
                    "mime_type": "text/plain",
                    "size_bytes": 100,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            {
                "id": "doc-2",
                "url": "https://example.com/2",
                "content": "Another test document with more content for testing purposes.",
                "metadata": {
                    "source": "test",
                    "mime_type": "text/plain",
                    "size_bytes": 150,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        ]

        for doc in documents:
            temp_store.store_document(doc)

        return temp_store

    def test_report_generation(self, sample_data, tmp_path):
        """Test that report generation works correctly."""
        generator = ReportGenerator(sample_data, str(tmp_path))

        # Generate report
        report_path = generator.generate_report()

        # Check that report file was created
        assert Path(report_path).exists()

        # Check report content
        with open(report_path) as f:
            content = f.read()

        assert "AI-Infant Weekly Report" in content
        assert "Tokens/Day" in content
        assert "Pages/Day" in content
        assert "Current Adapter" in content
        assert "Disk Usage" in content

    def test_job_logging(self, sample_data, tmp_path):
        """Test that report generation logs JobV1 entries."""
        generator = ReportGenerator(sample_data, str(tmp_path))

        # Generate report
        generator.generate_report()

        # Check that job was logged
        report_jobs = sample_data.get_jobs(job_type="report")
        assert len(report_jobs) >= 1

        # Check job details
        job = report_jobs[0]
        assert job["type"] == "report"
        assert job["status"] == "completed"
        assert "input" in job
        assert "output" in job

    def test_metrics_calculation(self, sample_data):
        """Test that metrics are calculated correctly."""
        generator = ReportGenerator(sample_data)

        # Test tokens per day calculation
        week_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        week_end = week_start + timedelta(days=7)

        tokens_metrics = generator._calculate_tokens_per_day(week_start, week_end)
        assert "tokens_per_day" in tokens_metrics
        assert "total_tokens" in tokens_metrics
        assert "documents_processed" in tokens_metrics
        assert tokens_metrics["documents_processed"] >= 0

    def test_eval_score_delta(self, sample_data):
        """Test evaluation score delta calculation."""
        generator = ReportGenerator(sample_data)

        eval_metrics = generator._get_eval_score_delta()
        assert "current_score" in eval_metrics
        assert "previous_score" in eval_metrics
        assert "delta" in eval_metrics
        assert "current_adapter_id" in eval_metrics
        assert "previous_adapter_id" in eval_metrics

    def test_disk_usage_calculation(self, sample_data):
        """Test disk usage calculation."""
        generator = ReportGenerator(sample_data)

        disk_usage = generator._calculate_disk_usage()
        assert "total_bytes" in disk_usage
        assert "total_mb" in disk_usage
        assert "files" in disk_usage
        assert isinstance(disk_usage["files"], list)


class TestRetentionManager:
    """Test the retention manager functionality."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        # Create a temporary directory for the database
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"

        store = Store(str(db_path))
        yield store
        store.close()

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_traces(self, temp_store):
        """Create sample traces for testing."""
        # Add sample traces with duplicates and varying scores
        traces = [
            # High-scoring trace
            {
                "id": "trace-1",
                "job_id": "job-1",
                "component": "crawl",
                "operation": "fetch",
                "status": "completed",
                "timestamp": "2025-08-19T20:00:00Z",
                "duration_ms": 100,
                "input": {"url": "https://example.com/1"},
                "output": {"status": "success"},
                "metadata": {"version": "0.1.0"},
            },
            # Duplicate of trace-1
            {
                "id": "trace-2",
                "job_id": "job-2",
                "component": "crawl",
                "operation": "fetch",
                "status": "completed",
                "timestamp": "2025-08-19T20:00:00Z",
                "duration_ms": 100,
                "input": {"url": "https://example.com/1"},
                "output": {"status": "success"},
                "metadata": {"version": "0.1.0"},
            },
            # Low-scoring trace (failed)
            {
                "id": "trace-3",
                "job_id": "job-3",
                "component": "crawl",
                "operation": "fetch",
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 5000,
                "input": {"url": "https://example.com/2"},
                "output": None,
                "error": {"message": "Connection failed"},
                "metadata": {"version": "0.1.0"},
            },
            # Another low-scoring trace (slow)
            {
                "id": "trace-4",
                "job_id": "job-4",
                "component": "text",
                "operation": "parse",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 15000,
                "input": {"content": "test"},
                "output": {"parsed": True},
                "metadata": {"version": "0.1.0"},
            },
            # Medium-scoring trace
            {
                "id": "trace-5",
                "job_id": "job-5",
                "component": "learn",
                "operation": "train",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 2000,
                "input": {"model": "test"},
                "output": {"accuracy": 0.8},
                "metadata": {"version": "0.1.0"},
            },
        ]

        for trace in traces:
            temp_store.store_trace(trace)

        return temp_store

    def test_duplicate_detection(self, sample_traces):
        """Test that duplicate traces are detected correctly."""
        manager = RetentionManager(sample_traces)

        duplicate_ids = manager._find_duplicate_traces()

        # Should find trace-2 as duplicate of trace-1
        assert "trace-2" in duplicate_ids
        assert "trace-1" not in duplicate_ids  # First occurrence is kept

    def test_low_scoring_detection(self, sample_traces):
        """Test that low-scoring traces are detected correctly."""
        manager = RetentionManager(sample_traces)

        # With 5 traces, bottom 25% = 1 trace
        low_scoring_ids = manager._find_low_scoring_traces(25.0)

        assert len(low_scoring_ids) == 1
        # Should be trace-3 (failed) or trace-4 (slow)
        assert low_scoring_ids[0] in ["trace-3", "trace-4"]

    def test_retention_execution(self, sample_traces):
        """Test that retention process works correctly."""
        manager = RetentionManager(sample_traces)

        initial_count = len(sample_traces.get_traces())

        # Run retention
        result = manager.run_retention(
            remove_duplicates=True, remove_low_scoring=True, low_scoring_percentile=25.0
        )

        # Check results
        assert result["traces_removed"] > 0
        assert result["duplicates_removed"] >= 1
        assert result["low_scoring_removed"] >= 0

        # Check final count
        final_count = len(sample_traces.get_traces())
        assert final_count < initial_count

    def test_job_logging(self, sample_traces):
        """Test that retention logs JobV1 entries."""
        manager = RetentionManager(sample_traces)

        # Run retention
        manager.run_retention()

        # Check that job was logged
        retention_jobs = sample_traces.get_jobs(job_type="retention")
        assert len(retention_jobs) >= 1

        # Check job details
        job = retention_jobs[0]
        assert job["type"] == "retention"
        assert job["status"] == "completed"
        assert "input" in job
        assert "output" in job

    def test_retention_stats(self, sample_traces):
        """Test retention statistics calculation."""
        manager = RetentionManager(sample_traces)

        stats = manager.get_retention_stats()

        assert "total_traces" in stats
        assert "duplicate_count" in stats
        assert "low_scoring_count" in stats
        assert "average_score" in stats
        assert "disk_usage_mb" in stats

        assert stats["total_traces"] == 5
        assert stats["duplicate_count"] >= 1

    def test_retention_report(self, sample_traces):
        """Test retention report generation."""
        manager = RetentionManager(sample_traces)

        report = manager.create_retention_report()

        assert "Retention Analysis Report" in report
        assert "Total Traces" in report
        assert "Duplicate Traces" in report
        assert "Low-Scoring Traces" in report

    def test_trace_scoring(self, sample_traces):
        """Test trace scoring algorithm."""
        traces = sample_traces.get_traces()

        # Test scoring for different types of traces
        for trace in traces:
            score = sample_traces.calculate_trace_score(trace)
            assert isinstance(score, float)
            assert score >= 0.0

            # Failed traces should have lower scores
            if trace["status"] == "failed":
                assert score < 1.0

            # Long duration traces should have lower scores
            if trace["duration_ms"] > 10000:
                assert score < 1.0

    def test_disk_usage_tracking(self, sample_traces):
        """Test disk usage tracking before and after retention."""
        manager = RetentionManager(sample_traces)

        # Get initial disk usage
        sample_traces.get_disk_usage()

        # Run retention
        result = manager.run_retention()

        # Get final disk usage
        sample_traces.get_disk_usage()

        # Check that disk usage is tracked
        assert "initial_disk_usage_mb" in result
        assert "final_disk_usage_mb" in result
        assert "disk_savings_mb" in result

        # Disk savings should be calculated
        assert result["disk_savings_mb"] == (
            result["initial_disk_usage_mb"] - result["final_disk_usage_mb"]
        )


class TestIntegration:
    """Integration tests for report and retention together."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        # Create a temporary directory for the database
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"

        store = Store(str(db_path))
        yield store
        store.close()

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_report_after_retention(self, temp_store, tmp_path):
        """Test that report works correctly after retention."""
        # Add some sample data
        traces = [
            {
                "id": "trace-1",
                "job_id": "job-1",
                "component": "crawl",
                "operation": "fetch",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 100,
                "input": {"url": "https://example.com"},
                "output": {"status": "success"},
                "metadata": {"version": "0.1.0"},
            },
            {
                "id": "trace-2",
                "job_id": "job-1",
                "component": "crawl",
                "operation": "fetch",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "duration_ms": 100,
                "input": {"url": "https://example.com"},
                "output": {"status": "success"},
                "metadata": {"version": "0.1.0"},
            },
        ]

        for trace in traces:
            temp_store.store_trace(trace)

        # Run retention
        retention_manager = RetentionManager(temp_store)
        retention_result = retention_manager.run_retention()

        # Generate report
        report_generator = ReportGenerator(temp_store, str(tmp_path))
        report_path = report_generator.generate_report()

        # Check that both operations completed successfully
        assert retention_result["traces_removed"] > 0
        assert Path(report_path).exists()

        # Check that both jobs were logged
        retention_jobs = temp_store.get_jobs(job_type="retention")
        report_jobs = temp_store.get_jobs(job_type="report")

        assert len(retention_jobs) >= 1
        assert len(report_jobs) >= 1

    def test_error_handling(self, temp_store):
        """Test error handling in both scripts."""
        # Test report with empty store
        generator = ReportGenerator(temp_store)

        # Should not raise an exception
        try:
            generator.generate_report()
        except Exception as e:
            pytest.fail(f"Report generation should not fail with empty store: {e}")

        # Test retention with empty store
        manager = RetentionManager(temp_store)

        # Should not raise an exception
        try:
            result = manager.run_retention()
            assert result["traces_removed"] == 0
        except Exception as e:
            pytest.fail(f"Retention should not fail with empty store: {e}")


if __name__ == "__main__":
    pytest.main([__file__])

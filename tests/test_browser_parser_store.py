"""Tests for browser, parser, and store components."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from bs4 import BeautifulSoup

from ai_infant.crawl import Browser
from ai_infant.data import Store
from ai_infant.text import ParsedDocument, Parser


class TestBrowser:
    """Test browser functionality."""

    @pytest.fixture
    def store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            yield Store(db_path)

    @pytest.fixture
    def browser(self, store):
        """Create a browser instance for testing."""
        # Mock the browser to avoid Playwright async/sync issues
        with patch("ai_infant.crawl.browser.sync_playwright"):
            return Browser(store)

    def test_browser_initialization(self, store):
        """Test browser initialization."""
        browser = Browser(store, user_agent="TestBot/1.0")
        assert browser.user_agent == "TestBot/1.0"
        assert browser.rate_limit_delay == 2.0  # Updated to match actual implementation
        assert browser.confidence_threshold == 0.7  # Default confidence threshold

    def test_robots_parser_caching(self, browser):
        """Test robots.txt parser caching."""
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"

        # Should use same parser for same domain
        parser1 = browser._get_robots_parser(url1)
        parser2 = browser._get_robots_parser(url2)
        assert parser1 is parser2

        # Different domain should get different parser
        url3 = "https://other.com/page"
        parser3 = browser._get_robots_parser(url3)
        assert parser3 is not parser1

    @patch("requests.Session.get")
    def test_fetch_success(self, mock_get, browser):
        """Test successful fetch operation."""
        # Mock response
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock robots.txt to allow access
        with patch.object(browser, "_can_fetch", return_value=True):
            # Mock browser page methods that the fetch method uses
            with patch.object(browser, "page") as mock_page:
                mock_page.content.return_value = (
                    "<html><body>Test content</body></html>"
                )
                mock_page.title.return_value = "Test Page"
                mock_page.query_selector.return_value = None  # No meta description

                # Mock viewport and response
                mock_page.viewport_size = {"width": 1920, "height": 1080}
                mock_response_obj = Mock()
                mock_response_obj.status = 200
                mock_response_obj.headers = {"content-type": "text/html"}
                mock_page.goto.return_value = mock_response_obj

                with patch.object(
                    browser, "_take_screenshot", return_value="/tmp/test.png"
                ):
                    result = browser.fetch("https://example.com/test")

        assert result is not None
        assert result.url == "https://example.com/test"
        assert result.content == "<html><body>Test content</body></html>"
        assert result.mime_type == "text/html"
        assert result.status_code == 200
        assert result.checksum is not None

    @patch("requests.Session.get")
    def test_fetch_robots_forbidden(self, mock_get, browser):
        """Test fetch when robots.txt forbids access."""
        with patch.object(browser, "_can_fetch", return_value=False):
            result = browser.fetch("https://example.com/forbidden")

        assert result is None
        mock_get.assert_not_called()

    @patch("requests.Session.get")
    def test_fetch_request_error(self, mock_get, browser):
        """Test fetch when request fails."""
        mock_get.side_effect = Exception("Network error")

        with patch.object(browser, "_can_fetch", return_value=True):
            result = browser.fetch("https://example.com/error")

        assert result is None

    def test_confidence_threshold_management(self, browser):
        """Test confidence threshold getter and setter."""
        # Test default threshold
        assert browser.get_confidence_threshold() == 0.7

        # Test setting valid threshold
        browser.set_confidence_threshold(0.8)
        assert browser.get_confidence_threshold() == 0.8

        # Test setting minimum threshold
        browser.set_confidence_threshold(0.0)
        assert browser.get_confidence_threshold() == 0.0

        # Test setting maximum threshold
        browser.set_confidence_threshold(1.0)
        assert browser.get_confidence_threshold() == 1.0

        # Test invalid threshold raises error
        with pytest.raises(ValueError):
            browser.set_confidence_threshold(-0.1)

        with pytest.raises(ValueError):
            browser.set_confidence_threshold(1.1)

    def test_execute_action_high_confidence(self, browser):
        """Test execute_action with high confidence."""
        # Mock page methods
        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"
                mock_click.return_value = True

                # Execute action with high confidence
                result = browser.execute_action("click", "#button", 0.9)

                # Should execute successfully
                assert result is True
                mock_click.assert_called_once_with("#button")

                # Check action history
                history = browser.get_action_history()
                assert len(history) == 1
                assert history[0]["action"] == "click"
                assert history[0]["confidence"] == 0.9
                assert (
                    "reason" not in history[0]
                )  # No skip reason for successful actions

    def test_execute_action_low_confidence(self, browser):
        """Test execute_action with low confidence."""
        # Mock page methods
        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"
                mock_click.return_value = True

                # Execute action with low confidence
                result = browser.execute_action("click", "#button", 0.5)

                # Should be skipped due to low confidence
                assert result is False
                mock_click.assert_not_called()

                # Check action history
                history = browser.get_action_history()
                assert len(history) == 1
                assert history[0]["action"] == "click"
                assert history[0]["confidence"] == 0.5
                assert history[0]["reason"] == "low_confidence"

    def test_execute_action_unknown_type(self, browser):
        """Test execute_action with unknown action type."""
        result = browser.execute_action("unknown_action", "#element", 0.9)
        assert result is False

    def test_execute_action_fill_without_value(self, browser):
        """Test execute_action fill type without value parameter."""
        result = browser.execute_action("fill", "#input", 0.9)
        assert result is False

    def test_execute_action_fill_with_value(self, browser):
        """Test execute_action fill type with value parameter."""
        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "fill_form") as mock_fill:
                mock_page.url = "https://example.com"
                mock_fill.return_value = True

                result = browser.execute_action(
                    "fill", "#input", 0.9, value="test value"
                )

                assert result is True
                mock_fill.assert_called_once_with({"input": "test value"})

    def test_click_element_with_low_confidence(self, browser):
        """Test click_element with low confidence parameter."""
        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com"
            mock_page.wait_for_selector.return_value = None  # Element not found

            # Click with low confidence
            result = browser.click_element("#button", confidence=0.5)

            # Should be skipped
            assert result is False

            # Check action history for skipped action
            history = browser.get_action_history()
            assert len(history) == 1
            assert history[0]["action"] == "click"
            assert history[0]["confidence"] == 0.5
            assert history[0]["reason"] == "low_confidence"

    def test_fill_form_with_low_confidence(self, browser):
        """Test fill_form with low confidence parameter."""
        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com"

            # Fill form with low confidence
            result = browser.fill_form({"username": "test"}, confidence=0.5)

            # Should be skipped
            assert result is False

            # Check action history for skipped action
            history = browser.get_action_history()
            assert len(history) == 1
            assert history[0]["action"] == "fill_form"
            assert history[0]["confidence"] == 0.5
            assert history[0]["reason"] == "low_confidence"

    def test_custom_confidence_threshold(self, browser):
        """Test custom confidence threshold affects action execution."""
        # Set custom threshold
        browser.set_confidence_threshold(0.9)

        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"
                mock_click.return_value = True

                # Test action with confidence below new threshold
                result = browser.execute_action("click", "#button", 0.8)
                assert result is False
                mock_click.assert_not_called()

                # Test action with confidence above new threshold
                result = browser.execute_action("click", "#button", 0.95)
                assert result is True
                mock_click.assert_called_once_with("#button")

    def test_real_world_confidence_scenarios(self, browser):
        """Test confidence validation with realistic scenarios."""
        # Scenario 1: High confidence actions should execute
        high_confidence_actions = [
            (0.9, "click", "#submit-button"),
            (0.95, "fill", "#search-input"),
            (0.85, "hover", ".menu-item"),
        ]

        with patch.object(browser, "page") as mock_page:
            with (
                patch.object(browser, "click_element") as mock_click,
                patch.object(browser, "fill_form") as mock_fill,
                patch.object(browser, "hover_element") as mock_hover,
            ):
                mock_page.url = "https://example.com"
                mock_click.return_value = True
                mock_fill.return_value = True
                mock_hover.return_value = True

                for confidence, action_type, selector in high_confidence_actions:
                    if action_type == "click":
                        result = browser.execute_action(
                            action_type, selector, confidence
                        )
                    elif action_type == "fill":
                        result = browser.execute_action(
                            action_type, selector, confidence, value="test"
                        )
                    elif action_type == "hover":
                        result = browser.execute_action(
                            action_type, selector, confidence
                        )

                    assert result is True, (
                        f"High confidence action {action_type} should execute"
                    )

                # Verify all actions were called
                assert mock_click.call_count == 1
                assert mock_fill.call_count == 1
                assert mock_hover.call_count == 1

                # Check action history
                history = browser.get_action_history()
                assert len(history) == 3
                for i, (confidence, action_type, _) in enumerate(
                    high_confidence_actions
                ):
                    assert history[i]["confidence"] == confidence
                    assert history[i]["action"] == action_type
                    assert "reason" not in history[i]

        # Clear action history for next scenario
        browser.clear_action_history()

        # Scenario 2: Low confidence actions should be rejected
        low_confidence_actions = [
            (0.5, "click", "#unreliable-button"),
            (0.3, "fill", "#uncertain-input"),
            (0.6, "select", "#questionable-dropdown"),
        ]

        with patch.object(browser, "page") as mock_page:
            with (
                patch.object(browser, "click_element") as mock_click,
                patch.object(browser, "fill_form") as mock_fill,
                patch.object(browser, "select_option") as mock_select,
            ):
                mock_page.url = "https://example.com"

                for confidence, action_type, selector in low_confidence_actions:
                    if action_type == "click":
                        result = browser.execute_action(
                            action_type, selector, confidence
                        )
                    elif action_type == "fill":
                        result = browser.execute_action(
                            action_type, selector, confidence, value="test"
                        )
                    elif action_type == "select":
                        result = browser.execute_action(
                            action_type, selector, confidence, value="option1"
                        )

                    assert result is False, (
                        f"Low confidence action {action_type} should be rejected"
                    )

                # Verify no actions were actually executed
                mock_click.assert_not_called()
                mock_fill.assert_not_called()
                mock_select.assert_not_called()

                # Check action history shows skipped actions
                history = browser.get_action_history()
                assert len(history) == 3
                for i, (confidence, action_type, _) in enumerate(
                    low_confidence_actions
                ):
                    assert history[i]["confidence"] == confidence
                    assert history[i]["action"] == action_type
                    assert history[i]["reason"] == "low_confidence"

    def test_confidence_threshold_edge_cases(self, browser):
        """Test edge cases for confidence threshold validation."""
        # Test exact threshold value (should execute)
        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"
                mock_click.return_value = True

                result = browser.execute_action("click", "#button", 0.7)
                assert result is True
                mock_click.assert_called_once_with("#button")

        browser.clear_action_history()

        # Test just below threshold (should be rejected)
        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"

                result = browser.execute_action("click", "#button", 0.69)
                assert result is False
                mock_click.assert_not_called()

                history = browser.get_action_history()
                assert len(history) == 1
                assert history[0]["reason"] == "low_confidence"

    def test_confidence_validation_performance(self, browser):
        """Test that confidence validation doesn't add significant overhead."""
        import time

        with patch.object(browser, "page") as mock_page:
            with patch.object(browser, "click_element") as mock_click:
                mock_page.url = "https://example.com"
                mock_click.return_value = True

                # Test multiple actions quickly
                start_time = time.time()
                for i in range(100):
                    result = browser.execute_action("click", f"#button-{i}", 0.9)
                    assert result is True
                end_time = time.time()

                # Should complete 100 actions in under 1 second (very generous limit)
                duration = end_time - start_time
                assert duration < 1.0, (
                    f"Confidence validation took too long: {duration:.3f}s"
                )

                # Verify all actions were executed
                assert mock_click.call_count == 100


class TestParser:
    """Test parser functionality."""

    @pytest.fixture
    def store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            yield Store(db_path)

    @pytest.fixture
    def parser(self, store):
        """Create a parser instance for testing."""
        return Parser(store)

    def test_quote_extraction(self, parser):
        """Test quote extraction from text."""
        text = 'This is a test with "a quoted phrase" and another "second quote".'
        quotes = parser._extract_quotes(text)

        assert len(quotes) == 2
        assert quotes[0]["text"] == "a quoted phrase"
        assert quotes[1]["text"] == "second quote"
        assert "context" in quotes[0]
        assert "position" in quotes[0]

    def test_html_cleaning(self, parser):
        """Test HTML cleaning functionality."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <script>alert('test');</script>
                <h1>Main Title</h1>
                <p>This is a paragraph with "quoted text".</p>
                <style>body { color: red; }</style>
            </body>
        </html>
        """

        clean_text = parser._clean_html(html)

        assert "alert('test')" not in clean_text
        assert "body { color: red; }" not in clean_text
        assert "Main Title" in clean_text
        assert "This is a paragraph" in clean_text

    def test_title_extraction(self, parser):
        """Test title extraction from HTML."""
        html = """
        <html>
            <head><title>Test Title</title></head>
            <body>Content</body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        title = parser._extract_title(soup)
        assert title == "Test Title"

    def test_author_extraction(self, parser):
        """Test author extraction from HTML."""
        html = """
        <html>
            <head>
                <meta name="author" content="John Doe">
            </head>
            <body>Content</body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        author = parser._extract_author(soup)
        assert author == "John Doe"

    def test_language_detection(self, parser):
        """Test language detection."""
        # Test with English text that has enough common words
        english_text = "The quick brown fox jumps over the lazy dog and the cat."
        language = parser._detect_language(english_text)
        assert language == "en"

        # Test with non-English text
        non_english_text = "Ceci est un texte en français."
        language = parser._detect_language(non_english_text)
        assert language is None  # Basic implementation doesn't detect French

        # Test with minimal English text that should be detected
        minimal_english = "The and or but in on at to for of with by."
        language = parser._detect_language(minimal_english)
        assert language == "en"

    def test_parse_html(self, parser):
        """Test HTML parsing."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a paragraph with "quoted text".</p>
            </body>
        </html>
        """

        result = parser.parse_html("https://example.com/test", html)

        assert result is not None
        assert result.url == "https://example.com/test"
        assert result.title == "Test Page"
        assert "Main Title" in result.content
        assert len(result.quotes) == 1
        assert result.quotes[0]["text"] == "quoted text"
        assert result.checksum is not None

    def test_parse_plain_text(self, parser):
        """Test plain text parsing."""
        text = 'This is plain text with "a quote".'

        result = parser.parse("https://example.com/text", text, "text/plain")

        assert result is not None
        assert result.url == "https://example.com/text"
        assert result.content == text
        assert len(result.quotes) == 1
        assert result.quotes[0]["text"] == "a quote"

    def test_parse_pdf_error_handling(self, parser):
        """Test PDF parsing error handling."""
        # Test with invalid PDF content
        invalid_pdf = b"Not a valid PDF file"

        result = parser.parse_pdf("https://example.com/invalid.pdf", invalid_pdf)

        # Should return None and log error, not crash
        assert result is None


class TestStore:
    """Test store functionality."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            yield Store(db_path)

    def test_store_initialization(self, temp_store):
        """Test store initialization."""
        assert temp_store.db_path.exists()
        assert temp_store.conn is not None

    def test_job_storage(self, temp_store):
        """Test job storage functionality."""
        job_data = {
            "id": "test-job-1",
            "type": "fetch",
            "status": "completed",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "input": {"url": "https://example.com"},
            "output": {"status_code": 200},
            "metadata": {"version": "0.1.0"},
        }

        job_id = temp_store.store_job(job_data)
        assert job_id == "test-job-1"

        # Check if job exists in database
        stored_job = temp_store.get_job("test-job-1")
        assert stored_job is not None
        assert stored_job["type"] == "fetch"
        assert stored_job["status"] == "completed"

    def test_document_storage_and_dedup(self, temp_store):
        """Test document storage with deduplication."""
        # Create a mock parsed document
        from datetime import datetime

        doc = ParsedDocument(
            url="https://example.com/test",
            content="Test content",
            title="Test Title",
            author="Test Author",
            language="en",
            quotes=[],
            checksum="test-checksum-123",
            parse_time=datetime.utcnow(),
        )

        # Store document
        doc_id = temp_store.store_document(doc)
        assert doc_id is not None

        # Try to store same document again (should be skipped)
        doc_id2 = temp_store.store_document(doc)
        assert doc_id2 is None

        # Check document exists
        stored_doc = temp_store.get_document_by_checksum("test-checksum-123")
        assert stored_doc is not None
        assert stored_doc["url"] == "https://example.com/test"
        assert stored_doc["content"] == "Test content"

    def test_job_statistics(self, temp_store):
        """Test job statistics functionality."""
        # Add some test jobs
        jobs = [
            {"id": "job-1", "type": "fetch", "status": "completed"},
            {"id": "job-2", "type": "parse", "status": "completed"},
            {"id": "job-3", "type": "fetch", "status": "failed"},
        ]

        for job in jobs:
            job_data = {
                **job,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
                "input": {},
                "output": {},
                "metadata": {},
            }
            temp_store.store_job(job_data)

        stats = temp_store.get_job_stats()

        assert stats["jobs_by_type"]["fetch"] == 2
        assert stats["jobs_by_type"]["parse"] == 1
        assert stats["jobs_by_status"]["completed"] == 2
        assert stats["jobs_by_status"]["failed"] == 1

    def test_context_manager(self):
        """Test store context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            store = Store(db_path)
            with store:
                assert store.conn is not None
            # Connection should be closed after context exit
            assert store.conn is None


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a complete pipeline for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            store = Store(db_path)
            with patch("ai_infant.crawl.browser.sync_playwright"):
                browser = Browser(store)
            parser = Parser(store)
            yield store, browser, parser

    @patch("requests.Session.get")
    def test_complete_pipeline(self, mock_get, pipeline):
        """Test complete fetch -> parse -> store pipeline."""
        store, browser, parser = pipeline

        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a paragraph with "quoted text".</p>
            </body>
        </html>
        """
        mock_response.headers = {"content-type": "text/html"}
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock robots.txt to allow access
        with patch.object(browser, "_can_fetch", return_value=True):
            # Fetch
            fetch_result = browser.fetch("https://example.com/test")
            assert fetch_result is not None

            # Parse
            parsed_doc = parser.parse(
                fetch_result.url, fetch_result.content, fetch_result.mime_type
            )
            assert parsed_doc is not None

            # Store
            doc_id = store.store_document(parsed_doc)
            assert doc_id is not None

            # Verify job logging
            stats = store.get_job_stats()
            assert stats["jobs_by_type"]["fetch"] == 1
            assert stats["jobs_by_type"]["parse"] == 1
            assert stats["document_count"] == 1

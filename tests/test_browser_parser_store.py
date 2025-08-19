"""Tests for browser, parser, and store components."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from bs4 import BeautifulSoup

from ai_infant.crawl import Browser, FetchResult
from ai_infant.data import Store
from ai_infant.text import Parser, ParsedDocument


class TestBrowser:
    """Test browser functionality."""
    
    @pytest.fixture
    def store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Store(temp_dir)
    
    @pytest.fixture
    def browser(self, store):
        """Create a browser instance for testing."""
        return Browser(store)
    
    def test_browser_initialization(self, store):
        """Test browser initialization."""
        browser = Browser(store, user_agent="TestBot/1.0")
        assert browser.user_agent == "TestBot/1.0"
        assert browser.rate_limit_delay == 1.0
        assert browser.session.headers["User-Agent"] == "TestBot/1.0"
    
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
    
    @patch('requests.Session.get')
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
        with patch.object(browser, '_can_fetch', return_value=True):
            result = browser.fetch("https://example.com/test")
        
        assert result is not None
        assert result.url == "https://example.com/test"
        assert result.content == "<html><body>Test content</body></html>"
        assert result.mime_type == "text/html"
        assert result.status_code == 200
        assert result.checksum is not None
    
    @patch('requests.Session.get')
    def test_fetch_robots_forbidden(self, mock_get, browser):
        """Test fetch when robots.txt forbids access."""
        with patch.object(browser, '_can_fetch', return_value=False):
            result = browser.fetch("https://example.com/forbidden")
        
        assert result is None
        mock_get.assert_not_called()
    
    @patch('requests.Session.get')
    def test_fetch_request_error(self, mock_get, browser):
        """Test fetch when request fails."""
        mock_get.side_effect = Exception("Network error")
        
        with patch.object(browser, '_can_fetch', return_value=True):
            result = browser.fetch("https://example.com/error")
        
        assert result is None


class TestParser:
    """Test parser functionality."""
    
    @pytest.fixture
    def store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Store(temp_dir)
    
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
        soup = BeautifulSoup(html, 'html.parser')
        
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
        soup = BeautifulSoup(html, 'html.parser')
        
        author = parser._extract_author(soup)
        assert author == "John Doe"
    
    def test_language_detection(self, parser):
        """Test language detection."""
        english_text = "The quick brown fox jumps over the lazy dog."
        language = parser._detect_language(english_text)
        assert language == "en"
        
        # Test with non-English text
        non_english_text = "Ceci est un texte en français."
        language = parser._detect_language(non_english_text)
        assert language is None  # Basic implementation doesn't detect French
    
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


class TestStore:
    """Test store functionality."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Store(temp_dir)
    
    def test_store_initialization(self, temp_store):
        """Test store initialization."""
        assert temp_store.data_dir.exists()
        assert temp_store.db_path.exists()
        assert temp_store.docs_file.exists()
        assert temp_store.jobs_file.exists()
    
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
            "metadata": {"version": "0.1.0"}
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
            parse_time=datetime.utcnow()
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
                "metadata": {}
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
            with Store(temp_dir) as store:
                assert store.conn is not None
            # Connection should be closed after context exit
            assert store.conn is None


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a complete pipeline for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Store(temp_dir)
            browser = Browser(store)
            parser = Parser(store)
            yield store, browser, parser
    
    @patch('requests.Session.get')
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
        with patch.object(browser, '_can_fetch', return_value=True):
            # Fetch
            fetch_result = browser.fetch("https://example.com/test")
            assert fetch_result is not None
            
            # Parse
            parsed_doc = parser.parse(fetch_result.url, fetch_result.content, fetch_result.mime_type)
            assert parsed_doc is not None
            
            # Store
            doc_id = store.store_document(parsed_doc)
            assert doc_id is not None
            
            # Verify job logging
            stats = store.get_job_stats()
            assert stats["jobs_by_type"]["fetch"] == 1
            assert stats["jobs_by_type"]["parse"] == 1
            assert stats["document_count"] == 1

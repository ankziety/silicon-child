"""Text parsing module for converting HTML/PDF to plaintext with anchored quotes."""

import hashlib
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Result of a parsing operation."""
    
    url: str
    content: str
    title: Optional[str]
    author: Optional[str]
    language: Optional[str]
    quotes: List[Dict[str, str]]
    checksum: str = Field(description="SHA-256 checksum of parsed content")
    parse_time: datetime


class Parser:
    """Parser for converting HTML/PDF content to plaintext with anchored quotes."""
    
    def __init__(self, store):
        """Initialize parser with storage."""
        self.store = store
        self.quote_patterns = [
            r'"([^"]{5,})"',  # Double quotes
            r"'([^']{5,})'",  # Single quotes
            r'«([^»]{5,})»',  # Guillemets
            r'„([^"]{5,})"',  # German quotes
            r'„([^"]{5,})"',  # Polish quotes
        ]
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract document title from HTML."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try meta tags
        meta_title = soup.find('meta', attrs={'name': 'title'})
        if meta_title:
            return meta_title.get('content', '').strip()
        
        # Try Open Graph title
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            return og_title.get('content', '').strip()
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract document author from HTML."""
        # Try meta author tag
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author:
            return meta_author.get('content', '').strip()
        
        # Try Open Graph author
        og_author = soup.find('meta', attrs={'property': 'article:author'})
        if og_author:
            return og_author.get('content', '').strip()
        
        # Try schema.org author
        schema_author = soup.find('meta', attrs={'property': 'schema:author'})
        if schema_author:
            return schema_author.get('content', '').strip()
        
        return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language from text content."""
        # Simple language detection based on common words
        # This is a basic implementation - could be enhanced with proper NLP
        text_lower = text.lower()
        
        # English indicators
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if english_count > 1:
            return 'en'
        
        # Could add more language detection logic here
        return None
    
    def _extract_quotes(self, text: str) -> List[Dict[str, str]]:
        """Extract and anchor quotes from text."""
        quotes = []
        
        for pattern in self.quote_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                quote_text = match.group(1).strip()
                if len(quote_text) >= 5:  # Minimum quote length
                    # Find context around the quote
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(text), match.end() + 100)
                    context = text[start_pos:end_pos].strip()
                    
                    quotes.append({
                        "text": quote_text,
                        "context": context,
                        "position": match.start(),
                        "pattern": pattern
                    })
        
        return quotes
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract plaintext."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _log_job(self, job_type: str, input_data: Dict[str, Any], 
                 output_data: Optional[Dict[str, Any]] = None,
                 error_data: Optional[Dict[str, Any]] = None) -> str:
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
                "timeout_seconds": 30
            }
        }
        
        self.store.store_job(job_data)
        return job_id
    
    def parse_html(self, url: str, html_content: str) -> Optional[ParsedDocument]:
        """Parse HTML content to plaintext with metadata and quotes."""
        start_time = datetime.utcnow()
        
        try:
            # Clean HTML and extract text
            clean_text = self._clean_html(html_content)
            
            # Parse HTML for metadata
            soup = BeautifulSoup(html_content, 'html.parser')
            title = self._extract_title(soup)
            author = self._extract_author(soup)
            language = self._detect_language(clean_text)
            
            # Extract quotes
            quotes = self._extract_quotes(clean_text)
            
            # Generate checksum
            checksum = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
            
            result = ParsedDocument(
                url=url,
                content=clean_text,
                title=title,
                author=author,
                language=language,
                quotes=quotes,
                checksum=checksum,
                parse_time=start_time
            )
            
            # Log successful job
            output_data = {
                "title": title,
                "author": author,
                "language": language,
                "quotes_count": len(quotes),
                "checksum": checksum,
                "content_length": len(clean_text)
            }
            self._log_job("parse", {"url": url, "content_type": "html"}, output_data)
            
            return result
            
        except Exception as e:
            error_data = {
                "type": "parse_error",
                "message": str(e),
                "stack": None
            }
            self._log_job("parse", {"url": url, "content_type": "html"}, error_data=error_data)
            return None
    
    def parse_pdf(self, url: str, pdf_content: bytes) -> Optional[ParsedDocument]:
        """Parse PDF content to plaintext with metadata and quotes."""
        # Note: This is a placeholder implementation
        # In a real implementation, you would use a PDF parsing library like PyPDF2 or pdfplumber
        start_time = datetime.utcnow()
        
        try:
            # For now, return None as PDF parsing requires additional dependencies
            # This would be implemented with proper PDF parsing library
            error_data = {
                "type": "pdf_not_supported",
                "message": "PDF parsing not yet implemented",
                "stack": None
            }
            self._log_job("parse", {"url": url, "content_type": "pdf"}, error_data=error_data)
            return None
            
        except Exception as e:
            error_data = {
                "type": "parse_error",
                "message": str(e),
                "stack": None
            }
            self._log_job("parse", {"url": url, "content_type": "pdf"}, error_data=error_data)
            return None
    
    def parse(self, url: str, content: str, mime_type: str) -> Optional[ParsedDocument]:
        """Parse content based on MIME type."""
        if 'text/html' in mime_type or 'application/xhtml+xml' in mime_type:
            return self.parse_html(url, content)
        elif 'application/pdf' in mime_type:
            return self.parse_pdf(url, content.encode('utf-8'))
        else:
            # For other text types, treat as plain text
            start_time = datetime.utcnow()
            
            try:
                clean_text = content.strip()
                quotes = self._extract_quotes(clean_text)
                checksum = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
                
                result = ParsedDocument(
                    url=url,
                    content=clean_text,
                    title=None,
                    author=None,
                    language=self._detect_language(clean_text),
                    quotes=quotes,
                    checksum=checksum,
                    parse_time=start_time
                )
                
                output_data = {
                    "quotes_count": len(quotes),
                    "checksum": checksum,
                    "content_length": len(clean_text)
                }
                self._log_job("parse", {"url": url, "content_type": mime_type}, output_data)
                
                return result
                
            except Exception as e:
                error_data = {
                    "type": "parse_error",
                    "message": str(e),
                    "stack": None
                }
                self._log_job("parse", {"url": url, "content_type": mime_type}, error_data=error_data)
                return None

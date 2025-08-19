"""Browser module for fetching web content with robots.txt compliance."""

import hashlib
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field


class FetchResult(BaseModel):
    """Result of a fetch operation."""
    
    url: str
    content: str
    mime_type: str
    size_bytes: int
    status_code: int
    headers: Dict[str, str]
    fetch_time: datetime
    checksum: str = Field(description="SHA-256 checksum of content")


class Browser:
    """Read-only browser with robots.txt compliance and rate limiting."""
    
    def __init__(self, store, user_agent: str = "AI-Infant/0.1.0"):
        """Initialize browser with storage and user agent."""
        self.store = store
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0.0
        self.robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
    
    def _get_robots_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        """Get or create robots.txt parser for a domain."""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if domain not in self.robots_cache:
            robots_url = f"{domain}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            
            try:
                parser.read()
                self.robots_cache[domain] = parser
            except Exception:
                # If robots.txt fails, create a permissive parser
                parser = urllib.robotparser.RobotFileParser()
                parser.allow_all = True
                self.robots_cache[domain] = parser
        
        return self.robots_cache[domain]
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        parser = self._get_robots_parser(url)
        return parser.can_fetch(self.user_agent, url)
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
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
    
    def fetch(self, url: str) -> Optional[FetchResult]:
        """Fetch content from URL with robots.txt compliance and rate limiting."""
        start_time = datetime.utcnow()
        
        # Check robots.txt
        if not self._can_fetch(url):
            error_data = {
                "type": "robots_forbidden",
                "message": f"URL {url} is forbidden by robots.txt",
                "stack": None
            }
            self._log_job("fetch", {"url": url}, error_data=error_data)
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            result = FetchResult(
                url=url,
                content=content,
                mime_type=response.headers.get('content-type', 'text/html'),
                size_bytes=len(content.encode('utf-8')),
                status_code=response.status_code,
                headers=dict(response.headers),
                fetch_time=start_time,
                checksum=checksum
            )
            
            # Log successful job
            output_data = {
                "status_code": response.status_code,
                "size_bytes": result.size_bytes,
                "checksum": checksum,
                "mime_type": result.mime_type
            }
            self._log_job("fetch", {"url": url}, output_data)
            
            return result
            
        except requests.RequestException as e:
            error_data = {
                "type": "request_error",
                "message": str(e),
                "stack": None
            }
            self._log_job("fetch", {"url": url}, error_data=error_data)
            return None
        except Exception as e:
            error_data = {
                "type": "unexpected_error",
                "message": str(e),
                "stack": None
            }
            self._log_job("fetch", {"url": url}, error_data=error_data)
            return None

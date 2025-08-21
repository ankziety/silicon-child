"""Browser crawling module for AI-Infant research agent."""

from .browser import Browser, FetchResult
from .browser_tool import create_browser_tool

__all__ = ["create_browser_tool", "Browser", "FetchResult"]

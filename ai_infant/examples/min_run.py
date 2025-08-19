#!/usr/bin/env python3
"""Minimal run example demonstrating document ingestion with job logging."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_infant.crawl import Browser
from ai_infant.data import Store
from ai_infant.text import Parser


def main():
    """Run minimal ingestion example."""
    print("AI-Infant: Minimal Run Example")
    print("=" * 40)

    # Initialize components
    store = Store("data")
    browser = Browser(store)
    parser = Parser(store)

    # Test URLs (using public, accessible pages)
    test_urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/robots.txt",
    ]

    ingested_count = 0
    skipped_count = 0

    for i, url in enumerate(test_urls, 1):
        print(f"\n{i}. Fetching: {url}")

        # Fetch content
        fetch_result = browser.fetch(url)
        if not fetch_result:
            print(f"   ❌ Failed to fetch {url}")
            continue

        print(f"   ✅ Fetched {fetch_result.size_bytes} bytes")

        # Parse content
        parsed_doc = parser.parse(url, fetch_result.content, fetch_result.mime_type)
        if not parsed_doc:
            print(f"   ❌ Failed to parse {url}")
            continue

        print(
            f"   ✅ Parsed content ({len(parsed_doc.content)} chars, {len(parsed_doc.quotes)} quotes)"
        )

        # Store document
        doc_id = store.store_document(parsed_doc)
        if doc_id:
            print(f"   ✅ Stored as {doc_id}")
            ingested_count += 1
        else:
            print("   ⏭️  Skipped (duplicate content)")
            skipped_count += 1

    # Print summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  Documents ingested: {ingested_count}")
    print(f"  Documents skipped: {skipped_count}")

    # Get job statistics
    stats = store.get_job_stats()
    print(f"  Jobs logged: {sum(stats.get('jobs_by_type', {}).values())}")

    print("\nJob breakdown:")
    for job_type, count in stats.get("jobs_by_type", {}).items():
        print(f"  {job_type}: {count}")

    print("\nStatus breakdown:")
    for status, count in stats.get("jobs_by_status", {}).items():
        print(f"  {status}: {count}")

    # Close store
    store.close()

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()

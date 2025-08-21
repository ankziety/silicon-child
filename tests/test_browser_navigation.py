"""Real browser navigation tests for confidence validation.

This module tests confidence validation with ACTUAL browser automation,
showing how it works in real web navigation scenarios.
"""

import time
from unittest.mock import Mock

import pytest
from playwright.sync_api import sync_playwright

from ai_infant.crawl.browser import Browser


class TestRealBrowserNavigation:
    """Test confidence validation with real browser automation."""

    @pytest.fixture(scope="function")
    def browser_instance(self):
        """Create a real browser instance for testing."""
        playwright = sync_playwright().start()
        # Allow running headful when env var HEADFUL_TESTS is set; default to headless
        headless = not bool("HEADFUL_TESTS" in __import__("os").environ)
        real_browser = playwright.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-web-security"],
        )
        page = real_browser.new_page(viewport={"width": 1920, "height": 1080})

        # Create our browser wrapper
        store = Mock()
        browser = Browser(store, headless=True)
        # Override the playwright instance with our real one
        browser.playwright = playwright
        browser.browser = real_browser
        browser.page = page

        # Reset retry state for clean test environment
        browser.reset_retry_state()

        yield browser

        # Cleanup
        try:
            page.close()
            real_browser.close()
            playwright.stop()
        except Exception:
            pass

    def test_real_navigation_with_confidence(self, browser_instance):
        """Test navigation to real websites with confidence validation."""
        browser = browser_instance

        # Test 1: Navigate to a simple, reliable website (high confidence)
        print("\nTesting high-confidence navigation...")
        result = browser.execute_action("navigate", "https://httpbin.org/html", 0.95)
        assert result is True, "High-confidence navigation should succeed"

        # Wait for page to load
        browser.page.wait_for_load_state("networkidle", timeout=5000)

        # Verify we're on the right page
        assert "httpbin.org" in browser.page.url
        title = browser.page.title()
        assert "Herman Melville" in title  # httpbin returns Moby Dick HTML

        # Check action history
        history = browser.get_action_history()
        assert len(history) == 1
        assert history[0]["action"] == "navigate"
        assert history[0]["confidence"] == 0.95
        assert "reason" not in history[0]

        print(f"Successfully navigated to: {browser.page.url}")
        print(f"Page title: {title}")

    def test_real_form_interaction_with_confidence(self, browser_instance):
        """Test real form filling with confidence validation."""
        browser = browser_instance

        # Navigate to httpbin forms page
        print("\nTesting form interaction...")
        nav_result = browser.execute_action(
            "navigate", "https://httpbin.org/forms/post", 0.9
        )
        assert nav_result is True

        browser.page.wait_for_load_state("networkidle", timeout=5000)

        # Try to fill form fields with different confidence levels
        fill_actions = [
            ("#custname", "John Doe", 0.85),  # High confidence
            ("#custtel", "555-0123", 0.82),  # High confidence
            ("#custemail", "john@example.com", 0.78),  # Medium confidence
            ("#delivery", "yes", 0.3),  # Low confidence - should be rejected
        ]

        successful_fills = 0
        for selector, value, confidence in fill_actions:
            result = browser.execute_action("fill", selector, confidence, value=value)
            if result:
                successful_fills += 1
                print(f"Filled {selector} with confidence {confidence}")
            else:
                print(f"Rejected {selector} with confidence {confidence}")

        # Should have filled 3 out of 4 fields (delivery rejected due to low confidence)
        assert successful_fills == 3, f"Expected 3 fills, got {successful_fills}"

        # Verify form data was actually filled
        name_field = browser.page.query_selector("#custname")
        assert name_field is not None
        assert name_field.input_value() == "John Doe"

        phone_field = browser.page.query_selector("#custtel")
        assert phone_field is not None
        assert phone_field.input_value() == "555-0123"

        # delivery field should still be empty (rejected due to low confidence)
        delivery_field = browser.page.query_selector("#delivery")
        if delivery_field:
            assert delivery_field.input_value() != "yes"

    def test_real_button_clicking_with_confidence(self, browser_instance):
        """Test real button clicking with confidence validation."""
        browser = browser_instance

        # Navigate to a page with buttons
        print("\nTesting button clicking...")
        browser.execute_action("navigate", "https://httpbin.org/html", 0.9)
        browser.page.wait_for_load_state("networkidle", timeout=5000)

        # Try to click a non-existent button (low confidence)
        click_result = browser.execute_action("click", "#non-existent-button", 0.4)
        assert click_result is False, (
            "Low-confidence click on non-existent element should be rejected"
        )

        # Try to click a button that doesn't exist (medium confidence)
        click_result = browser.execute_action("click", "#fake-button", 0.6)
        assert click_result is False, "Click on non-existent element should fail"

        # Try to click a valid element (high confidence)
        # First, let's create a simple test page by setting HTML content
        browser.page.set_content(
            """
        <html>
        <body>
            <button id="test-button" onclick="document.body.style.background='red'">
                Test Button
            </button>
            <div id="status">Not clicked</div>
            <script>
                document.getElementById('test-button').addEventListener('click', function() {
                    document.getElementById('status').textContent = 'Clicked!';
                });
            </script>
        </body>
        </html>
        """
        )

        # Now try to click the real button with high confidence
        click_result = browser.execute_action("click", "#test-button", 0.88)
        assert click_result is True, (
            "High-confidence click on real element should succeed"
        )

        # Wait a bit for JavaScript to execute
        browser.page.wait_for_timeout(100)

        # Verify the click actually worked
        status_element = browser.page.query_selector("#status")
        assert status_element is not None
        assert status_element.text_content() == "Clicked!"

        print("Button click successfully executed and JavaScript triggered")

    def test_confidence_threshold_adjustment_real_scenario(self, browser_instance):
        """Test confidence threshold adjustment with real browser interactions."""
        browser = browser_instance

        # Set a very high threshold (stricter)
        browser.set_confidence_threshold(0.9)

        # Navigate to test page
        browser.execute_action(
            "navigate", "data:text/html,<button id='btn'>Click me</button>", 0.95
        )

        # Try actions with different confidence levels
        actions = [
            (0.95, "#btn", True, "Should execute (above threshold)"),
            (0.85, "#btn", False, "Should be rejected (below threshold)"),
            (0.88, "#btn", False, "Should be rejected (below threshold)"),
            (0.92, "#btn", True, "Should execute (above threshold)"),
        ]

        results = []
        for confidence, selector, expected, description in actions:
            result = browser.execute_action("click", selector, confidence)
            results.append(result)
            print(
                f"Confidence {confidence}: {'Executed' if result else 'Rejected'} - {description}"
            )
            assert result is expected, f"Failed: {description}"

        # Verify results match expectations
        expected_results = [True, False, False, True]
        assert results == expected_results

        # Now lower the threshold and try again
        browser.set_confidence_threshold(0.8)
        print(f"\n🔄 Lowered threshold to: {browser.get_confidence_threshold()}")

        # Previously rejected action should now execute
        result = browser.execute_action("click", "#btn", 0.85)
        assert result is True, (
            "Previously rejected action should now execute with lower threshold"
        )

        print("✅ Confidence threshold adjustment working correctly")

    def test_error_handling_with_real_browser(self, browser_instance):
        """Test error handling with real browser failures."""
        browser = browser_instance

        # Try to navigate to a non-existent domain (high confidence but will fail)
        result = browser.execute_action(
            "navigate", "https://this-domain-definitely-does-not-exist-12345.com", 0.9
        )

        # The action should be attempted (confidence passed) but ultimately fail
        # This tests that confidence validation doesn't prevent legitimate attempts
        assert result is False, "Navigation to non-existent domain should fail"

        # Check that it was logged as attempted (no rejection reason)
        history = browser.get_action_history()
        navigate_action = history[-1]  # Last action
        assert navigate_action["action"] == "navigate"
        assert navigate_action["confidence"] == 0.9
        assert "reason" not in navigate_action  # Not rejected due to confidence

        print(
            "✅ Error handling working - confidence validation allows attempts, handles failures gracefully"
        )

    def test_performance_with_real_browser(self, browser_instance):
        """Test performance of confidence validation with real browser operations."""
        browser = browser_instance

        # Set up a simple test page
        browser.page.set_content(
            """
        <html><body>
        <button id="btn1">Button 1</button>
        <button id="btn2">Button 2</button>
        <button id="btn3">Button 3</button>
        </body></html>
        """
        )

        # Test multiple actions with mixed confidence levels
        actions = [
            ("click", "#btn1", 0.9),  # Execute
            ("click", "#btn2", 0.6),  # Reject
            ("click", "#btn3", 0.8),  # Execute
            ("click", "#nonexistent", 0.7),  # Reject (element doesn't exist)
        ]

        start_time = time.time()

        for action_type, selector, confidence in actions:
            browser.execute_action(action_type, selector, confidence)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly even with real browser operations
        assert duration < 5.0, f"Real browser operations took too long: {duration:.2f}s"

        # Verify correct execution/rejection
        history = browser.get_action_history()
        recent_actions = history[-4:]  # Last 4 actions

        executed = sum(1 for action in recent_actions if "reason" not in action)
        rejected = sum(
            1 for action in recent_actions if action.get("reason") == "low_confidence"
        )

        assert executed == 2, f"Expected 2 executions, got {executed}"
        assert rejected == 2, f"Expected 2 rejections, got {rejected}"

        print(f"✅ Performance test passed: {len(actions)} actions in {duration:.2f}s")
        print(f"   Executed: {executed}, Rejected: {rejected}")

    def test_confidence_validation_with_dynamic_content(self, browser_instance):
        """Test confidence validation with dynamically loaded content."""
        browser = browser_instance

        # Set up a page that loads content dynamically
        browser.page.set_content(
            """
        <html><body>
        <button id="load-btn">Load Content</button>
        <div id="dynamic-area"></div>
        <script>
            document.getElementById('load-btn').addEventListener('click', function() {
                setTimeout(function() {
                    document.getElementById('dynamic-area').innerHTML =
                        '<button id="dynamic-btn">Dynamic Button</button>';
                }, 500);
            });
        </script>
        </body></html>
        """
        )

        # Try to click the dynamic button before it's loaded (low confidence)
        result = browser.execute_action("click", "#dynamic-btn", 0.4)
        assert result is False, (
            "Should reject low-confidence action on non-existent element"
        )

        # Load the dynamic content
        browser.execute_action("click", "#load-btn", 0.9)
        browser.page.wait_for_timeout(600)  # Wait for dynamic content

        # Now try to click the dynamic button with high confidence
        result = browser.execute_action("click", "#dynamic-btn", 0.85)
        assert result is True, "Should execute high-confidence action on loaded element"

        print("✅ Dynamic content handling working correctly")
        print("   - Low confidence action on missing element: Rejected")
        print("   - High confidence action on loaded element: Executed")

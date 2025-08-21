"""Realistic integration tests for browser confidence validation.

This module tests confidence validation with more realistic browser automation
scenarios that simulate actual web page interactions.
"""

from unittest.mock import Mock, patch

import pytest

from ai_infant.crawl.browser import Browser


class TestRealisticBrowserAutomation:
    """Test confidence validation in realistic browser automation scenarios."""

    @pytest.fixture
    def browser(self):
        """Create browser with mocked Playwright."""
        with patch("ai_infant.crawl.browser.sync_playwright"):
            store = Mock()
            return Browser(store)

    def test_form_filling_workflow(self, browser):
        """Test realistic form filling workflow with confidence validation."""
        # Simulate a login form scenario
        form_actions = [
            {
                "action": "fill",
                "selector": "#username",
                "confidence": 0.95,
                "value": "testuser",
                "description": "Fill username field",
            },
            {
                "action": "fill",
                "selector": "#password",
                "confidence": 0.93,
                "value": "testpass",
                "description": "Fill password field",
            },
            {
                "action": "click",
                "selector": "#login-btn",
                "confidence": 0.88,
                "description": "Click login button",
            },
            {
                "action": "click",
                "selector": ".captcha-checkbox",
                "confidence": 0.45,  # Low confidence - should be rejected
                "description": "Click captcha (uncertain detection)",
            },
        ]

        with patch.object(browser, "page") as mock_page:
            # Mock successful element interactions
            mock_element = Mock()
            mock_element.is_visible.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.wait_for_selector.return_value = mock_element

            mock_page.url = "https://example.com/login"
            mock_page.wait_for_selector.return_value = mock_element

            with (
                patch.object(browser, "click_element") as mock_click,
                patch.object(browser, "fill_form") as mock_fill,
            ):
                mock_click.return_value = True
                mock_fill.return_value = True

                results = []
                for action in form_actions:
                    if action["action"] == "fill":
                        result = browser.execute_action(
                            action["action"],
                            action["selector"],
                            action["confidence"],
                            value=action["value"],
                        )
                    else:
                        result = browser.execute_action(
                            action["action"], action["selector"], action["confidence"]
                        )
                    results.append(result)

                # Verify results
                assert results[0] is True  # High confidence fill
                assert results[1] is True  # High confidence fill
                assert results[2] is True  # High confidence click
                assert results[3] is False  # Low confidence click - rejected

                # Verify methods were called correctly
                assert mock_fill.call_count == 2  # Only high confidence fills
                assert mock_click.call_count == 1  # Only high confidence click

                # Check action history
                history = browser.get_action_history()
                assert len(history) == 4

                # Verify low confidence action was logged with rejection reason
                low_confidence_action = history[3]
                assert low_confidence_action["confidence"] == 0.45
                assert low_confidence_action["reason"] == "low_confidence"

    def test_dynamic_content_handling(self, browser):
        """Test confidence validation with dynamic content scenarios."""
        # Simulate clicking on dynamically loaded elements
        dynamic_actions = [
            {
                "selector": "#initial-button",
                "confidence": 0.92,
                "expected_result": True,
                "description": "Click initial button (high confidence)",
            },
            {
                "selector": ".dynamically-added",
                "confidence": 0.78,
                "expected_result": True,
                "description": "Click dynamically added element (medium confidence)",
            },
            {
                "selector": ".maybe-not-loaded",
                "confidence": 0.52,
                "expected_result": False,
                "description": "Click potentially missing element (low confidence)",
            },
        ]

        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com/dynamic"

            # Mock different element availability scenarios
            def mock_wait_for_selector(selector, timeout=10000):
                if "maybe-not-loaded" in selector:
                    return None  # Simulate element not found
                mock_element = Mock()
                mock_element.is_visible.return_value = True
                mock_element.is_enabled.return_value = True
                return mock_element

            mock_page.wait_for_selector.side_effect = mock_wait_for_selector

            with patch.object(browser, "click_element") as mock_click:
                mock_click.return_value = True

                for action in dynamic_actions:
                    result = browser.execute_action(
                        "click", action["selector"], action["confidence"]
                    )

                    assert result is action["expected_result"], (
                        f"Failed for {action['description']}"
                    )

                # Verify click was only called for available elements
                assert mock_click.call_count == 2

    def test_navigation_and_interaction_flow(self, browser):
        """Test realistic navigation and interaction flow."""
        workflow = [
            ("navigate", "https://example.com", 0.98, True),
            ("wait", "#content", 0.95, True),
            ("click", "#menu-toggle", 0.87, True),
            ("fill", "#search", 0.82, True),
            ("click", "#search-btn", 0.76, True),  # Just below threshold
            ("hover", ".tooltip-trigger", 0.91, True),
            ("click", ".uncertain-element", 0.68, False),  # Should be rejected
        ]

        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com"
            mock_page.wait_for_selector.return_value = Mock()
            mock_page.goto.return_value = Mock(status=200, headers={})

            with (
                patch.object(browser, "click_element") as mock_click,
                patch.object(browser, "fill_form") as mock_fill,
                patch.object(browser, "hover_element") as mock_hover,
                patch.object(browser, "wait_for_element") as mock_wait,
            ):
                mock_click.return_value = True
                mock_fill.return_value = True
                mock_hover.return_value = True
                mock_wait.return_value = True

                executed_actions = []
                for action_type, selector, confidence, expected in workflow:
                    if action_type == "navigate":
                        result = browser.navigate_to(selector)
                    elif action_type == "wait":
                        result = browser.wait_for_element(selector)
                    elif action_type == "fill":
                        result = browser.execute_action(
                            action_type, selector, confidence, value="test"
                        )
                    else:
                        result = browser.execute_action(
                            action_type, selector, confidence
                        )

                    assert result is expected, (
                        f"Action {action_type} with confidence {confidence} failed expectation"
                    )

                    if result:
                        executed_actions.append(action_type)

                # Verify only high-confidence actions were executed
                assert "click" in executed_actions  # 0.87 confidence
                assert "fill" in executed_actions  # 0.82 confidence
                assert "hover" in executed_actions  # 0.91 confidence

                # Verify method call counts
                assert mock_click.call_count == 2  # menu-toggle and search-btn
                assert mock_fill.call_count == 1  # search field
                assert mock_hover.call_count == 1  # tooltip

    def test_error_recovery_scenarios(self, browser):
        """Test confidence validation during error recovery scenarios."""
        error_scenarios = [
            {
                "action": "click",
                "selector": "#flaky-button",
                "confidence": 0.89,
                "simulate_error": True,
                "expected_execution": False,
                "description": "Click flaky element that fails",
            },
            {
                "action": "fill",
                "selector": "#reliable-input",
                "confidence": 0.94,
                "simulate_error": False,
                "expected_execution": True,
                "description": "Fill reliable input that succeeds",
            },
        ]

        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com"

            for scenario in error_scenarios:
                browser.clear_action_history()

                with (
                    patch.object(browser, "click_element") as mock_click,
                    patch.object(browser, "fill_form") as mock_fill,
                ):
                    if scenario["action"] == "click":
                        mock_click.return_value = not scenario["simulate_error"]
                        result = browser.execute_action(
                            scenario["action"],
                            scenario["selector"],
                            scenario["confidence"],
                        )
                    else:
                        mock_fill.return_value = not scenario["simulate_error"]
                        result = browser.execute_action(
                            scenario["action"],
                            scenario["selector"],
                            scenario["confidence"],
                            value="test",
                        )

                    assert result is scenario["expected_execution"], (
                        f"Scenario failed: {scenario['description']}"
                    )

                    # Verify action was logged regardless of execution success
                    history = browser.get_action_history()
                    assert len(history) == 1
                    assert history[0]["action"] == scenario["action"]
                    assert history[0]["confidence"] == scenario["confidence"]

                    if scenario["simulate_error"]:
                        # Should still be logged as executed (confidence passed)
                        # but the underlying action failed
                        assert "reason" not in history[0]

    def test_performance_under_load(self, browser):
        """Test confidence validation performance with many actions."""
        import time

        # Simulate a complex workflow with many actions
        actions = [("click", f"#button-{i}", 0.8 + (i % 3) * 0.1) for i in range(50)]

        with patch.object(browser, "page") as mock_page:
            mock_page.url = "https://example.com"

            with patch.object(browser, "click_element") as mock_click:
                mock_click.return_value = True

                start_time = time.time()

                for action_type, selector, confidence in actions:
                    browser.execute_action(action_type, selector, confidence)

                end_time = time.time()
                duration = end_time - start_time

                # Should handle 50 actions quickly (under 2 seconds with mocks)
                assert duration < 2.0, f"Performance test failed: {duration:.3f}s"

                # Verify correct number of actions were executed vs rejected
                history = browser.get_action_history()
                executed = sum(1 for h in history if "reason" not in h)
                rejected = sum(
                    1 for h in history if h.get("reason") == "low_confidence"
                )

                # With threshold 0.7, actions with confidence 0.8 and 0.9 should execute
                # Actions with confidence 0.8 should be rejected (exactly at threshold)
                expected_executed = sum(1 for _, _, conf in actions if conf > 0.7)
                expected_rejected = sum(1 for _, _, conf in actions if conf <= 0.7)

                assert executed == expected_executed
                assert rejected == expected_rejected

"""Test package initialization and basic functionality."""

import ai_infant


def test_package_version() -> None:
    """Test that package version is defined."""
    assert hasattr(ai_infant, "__version__")
    assert ai_infant.__version__ == "0.1.0"


def test_package_author() -> None:
    """Test that package author is defined."""
    assert hasattr(ai_infant, "__author__")
    assert ai_infant.__author__ == "AI-Infant Team"


def test_package_docstring() -> None:
    """Test that package has a docstring."""
    assert ai_infant.__doc__ is not None
    assert "AI-Infant" in ai_infant.__doc__

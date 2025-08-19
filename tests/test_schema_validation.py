"""Test suite for validating fixtures against JSON schemas."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest


class TestSchemaValidation:
    """Test that all fixtures validate against their respective schemas."""

    @pytest.fixture(scope="class")
    def schemas_dir(self) -> Path:
        """Return the schemas directory path."""
        return Path(__file__).parent.parent / "schemas"

    @pytest.fixture(scope="class")
    def fixtures_dir(self) -> Path:
        """Return the fixtures directory path."""
        return Path(__file__).parent / "fixtures"

    def load_schema(self, schema_path: Path) -> dict[str, Any]:
        """Load a JSON schema from file."""
        with open(schema_path) as f:
            return dict[str, Any](json.load(f))

    def load_fixture(self, fixture_path: Path) -> dict[str, Any]:
        """Load a JSON fixture from file."""
        with open(fixture_path) as f:
            return dict[str, Any](json.load(f))

    def test_doc_v1_schema_valid(self, schemas_dir: Path) -> None:
        """Test that DocV1 schema is valid JSON Schema."""
        schema_path = schemas_dir / "doc.v1.json"
        schema = self.load_schema(schema_path)

        # Basic validation that it's a valid JSON Schema
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "DocV1"

    def test_trace_v1_schema_valid(self, schemas_dir: Path) -> None:
        """Test that TraceV1 schema is valid JSON Schema."""
        schema_path = schemas_dir / "trace.v1.json"
        schema = self.load_schema(schema_path)

        # Basic validation that it's a valid JSON Schema
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "TraceV1"

    def test_job_v1_schema_valid(self, schemas_dir: Path) -> None:
        """Test that JobV1 schema is valid JSON Schema."""
        schema_path = schemas_dir / "job.v1.json"
        schema = self.load_schema(schema_path)

        # Basic validation that it's a valid JSON Schema
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "JobV1"

    def test_doc_v1_fixture_validates(
        self, schemas_dir: Path, fixtures_dir: Path
    ) -> None:
        """Test that DocV1 fixture validates against its schema."""
        schema_path = schemas_dir / "doc.v1.json"
        fixture_path = fixtures_dir / "doc_v1_example.json"

        schema = self.load_schema(schema_path)
        fixture = self.load_fixture(fixture_path)

        # Validate fixture against schema
        jsonschema.validate(fixture, schema)

    def test_trace_v1_fixture_validates(
        self, schemas_dir: Path, fixtures_dir: Path
    ) -> None:
        """Test that TraceV1 fixture validates against its schema."""
        schema_path = schemas_dir / "trace.v1.json"
        fixture_path = fixtures_dir / "trace_v1_example.json"

        schema = self.load_schema(schema_path)
        fixture = self.load_fixture(fixture_path)

        # Validate fixture against schema
        jsonschema.validate(fixture, schema)

    def test_job_v1_fixture_validates(
        self, schemas_dir: Path, fixtures_dir: Path
    ) -> None:
        """Test that JobV1 fixture validates against its schema."""
        schema_path = schemas_dir / "job.v1.json"
        fixture_path = fixtures_dir / "job_v1_example.json"

        schema = self.load_schema(schema_path)
        fixture = self.load_fixture(fixture_path)

        # Validate fixture against schema
        jsonschema.validate(fixture, schema)

    def test_all_schemas_have_required_fields(self, schemas_dir: Path) -> None:
        """Test that all schemas have required metadata."""
        schemas = ["doc.v1.json", "trace.v1.json", "job.v1.json"]

        for schema_file in schemas:
            schema_path = schemas_dir / schema_file
            schema = self.load_schema(schema_path)

            # Check required schema metadata
            assert "$schema" in schema
            assert "$id" in schema
            assert "title" in schema
            assert "description" in schema
            assert "type" in schema
            assert "properties" in schema

    def test_schema_ids_are_valid_urls(self, schemas_dir: Path) -> None:
        """Test that schema $id fields are valid URLs."""
        schemas = ["doc.v1.json", "trace.v1.json", "job.v1.json"]

        for schema_file in schemas:
            schema_path = schemas_dir / schema_file
            schema = self.load_schema(schema_path)

            # Check that $id is a valid URL
            assert schema["$id"].startswith("https://ai-infant.dev/schemas/")
            assert schema["$id"].endswith(schema_file)

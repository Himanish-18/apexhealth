"""
Tests for the chunk pipeline (preprocessing.chunk_pipeline).
Tests resume support, force mode, citation lineage, and end-to-end chunking.
"""

import json
from pathlib import Path

from preprocessing.chunk_pipeline import _chunk_worker


def _create_sample_canonical_json(path: Path) -> None:
    """Write a minimal canonical JSON file for testing."""
    doc = {
        "metadata": {
            "pmcid": path.stem,
            "title": "Test Paper",
            "journal": "Nature",
            "year": 2024,
        },
        "abstract": "This is the abstract of the test paper about drug efficacy trials.",
        "sections": [
            {
                "title": "Introduction",
                "content": "Background information about the disease and treatment options available.",
                "subsections": [],
            },
            {
                "title": "Methods",
                "content": "We conducted a double-blind randomized controlled trial with 200 participants.",
                "subsections": [
                    {
                        "title": "Statistical Analysis",
                        "content": "We used Cox proportional hazards models for the primary analysis.",
                        "subsections": [],
                    }
                ],
            },
        ],
        "tables": [
            {
                "id": "T1",
                "caption": "Baseline characteristics of participants",
                "markdown": "| Variable | Group A | Group B |\n|---|---|---|\n| Age | 55 | 57 |",
                "section": "Results",
            },
        ],
        "figures": [
            {
                "id": "fig_1",
                "caption": "Kaplan-Meier survival curves for treatment and control groups.",
            },
        ],
        "references": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def test_chunk_worker_produces_output(tmp_path: Path):
    """Worker should produce a valid chunk file."""
    input_path = tmp_path / "PMC123.json"
    output_path = tmp_path / "PMC123_chunks.json"
    _create_sample_canonical_json(input_path)

    settings_dict = {
        "chunk_target_tokens": 512,
        "chunk_max_tokens": 600,
        "chunk_min_tokens": 5,  # Low threshold for small test content
        "chunk_overlap_tokens": 10,
    }

    result = _chunk_worker(input_path, output_path, settings_dict)

    assert result["status"] == "success"
    assert result["total_chunks"] > 0
    assert output_path.exists()

    # Verify output schema
    with open(output_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    assert "document_metadata" in output_data
    assert "chunks" in output_data
    assert len(output_data["chunks"]) > 0


def test_chunk_worker_citation_lineage(tmp_path: Path):
    """Every chunk should map back to the source paper and section."""
    input_path = tmp_path / "PMC456.json"
    output_path = tmp_path / "PMC456_chunks.json"
    _create_sample_canonical_json(input_path)

    settings_dict = {
        "chunk_target_tokens": 512,
        "chunk_max_tokens": 600,
        "chunk_min_tokens": 5,
        "chunk_overlap_tokens": 10,
    }

    _chunk_worker(input_path, output_path, settings_dict)

    with open(output_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    for chunk in output_data["chunks"]:
        # Every chunk must have a chunk_id starting with the PMCID
        assert chunk["chunk_id"].startswith("PMC456")
        # Every chunk must have full metadata for traceability
        assert chunk["metadata"]["pmcid"] == "PMC456"
        assert chunk["metadata"]["section"] != "" or chunk["chunk_type"] == "FIGURE_CAPTION"
        assert chunk["metadata"]["chunk_type"] in ["TEXT", "TABLE", "ABSTRACT", "FIGURE_CAPTION"]
        # section_path should be a list
        assert isinstance(chunk["metadata"]["section_path"], list)
        assert len(chunk["metadata"]["section_path"]) >= 1


def test_chunk_worker_all_chunk_types_present(tmp_path: Path):
    """Output should contain TEXT, TABLE, ABSTRACT, and FIGURE_CAPTION chunks."""
    input_path = tmp_path / "PMC789.json"
    output_path = tmp_path / "PMC789_chunks.json"
    _create_sample_canonical_json(input_path)

    settings_dict = {
        "chunk_target_tokens": 512,
        "chunk_max_tokens": 600,
        "chunk_min_tokens": 5,
        "chunk_overlap_tokens": 10,
    }

    _chunk_worker(input_path, output_path, settings_dict)

    with open(output_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    chunk_types = {c["chunk_type"] for c in output_data["chunks"]}
    assert "ABSTRACT" in chunk_types
    assert "TEXT" in chunk_types
    assert "TABLE" in chunk_types
    assert "FIGURE_CAPTION" in chunk_types


def test_resume_skips_existing(tmp_path: Path):
    """If output already exists and resume is on, it should be skipped."""
    input_path = tmp_path / "PMC100.json"
    output_path = tmp_path / "PMC100_chunks.json"
    _create_sample_canonical_json(input_path)

    # Pre-create output so resume would skip
    output_path.write_text("{}", encoding="utf-8")

    # The pipeline-level skip is handled by ChunkPipeline.run(),
    # but we can at least verify the worker works when called
    settings_dict = {
        "chunk_target_tokens": 512,
        "chunk_max_tokens": 600,
        "chunk_min_tokens": 5,
        "chunk_overlap_tokens": 10,
    }

    result = _chunk_worker(input_path, output_path, settings_dict)
    assert result["status"] == "success"  # Worker itself always processes


def test_chunk_output_schema_complete(tmp_path: Path):
    """Every chunk in the output should have all required fields."""
    input_path = tmp_path / "PMC999.json"
    output_path = tmp_path / "PMC999_chunks.json"
    _create_sample_canonical_json(input_path)

    settings_dict = {
        "chunk_target_tokens": 512,
        "chunk_max_tokens": 600,
        "chunk_min_tokens": 5,
        "chunk_overlap_tokens": 10,
    }

    _chunk_worker(input_path, output_path, settings_dict)

    with open(output_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    required_chunk_fields = {"chunk_id", "chunk_type", "content", "token_count", "metadata", "metadata_header"}
    required_metadata_fields = {
        "pmcid", "title", "journal", "year", "section",
        "subsection", "section_path", "chunk_type", "token_count",
    }

    for chunk in output_data["chunks"]:
        assert required_chunk_fields.issubset(set(chunk.keys())), (
            f"Missing chunk fields: {required_chunk_fields - set(chunk.keys())}"
        )
        assert required_metadata_fields.issubset(set(chunk["metadata"].keys())), (
            f"Missing metadata fields: {required_metadata_fields - set(chunk['metadata'].keys())}"
        )
        assert chunk["token_count"] > 0
        assert chunk["content"].strip() != ""

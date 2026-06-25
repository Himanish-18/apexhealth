"""
Healthcare Knowledge Navigator — Chunk Pipeline.

Orchestrates multiprocessing for batch chunking of canonical JSON files.
Mirrors the ParserPipeline pattern from Phase 2 for consistency.
Provides structured logging, resume support, and aggregate statistics.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from configs.settings import Settings
from preprocessing.chunker import ChunkedDocument
from preprocessing.chunk_validator import ChunkValidator
from preprocessing.semantic_chunker import SemanticChunker
from preprocessing.table_chunker import TableChunker

logger = logging.getLogger(__name__)


@dataclass
class ChunkStats:
    """Aggregate statistics for a chunking run."""

    total_documents: int = 0
    total_chunks: int = 0
    avg_chunk_size: float = 0.0
    largest_chunk: int = 0
    smallest_chunk: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    figure_chunks: int = 0
    abstract_chunks: int = 0
    rejected_chunks: int = 0
    skipped: int = 0
    failed: int = 0


def _chunk_worker(
    json_path: Path,
    output_path: Path,
    settings_dict: dict[str, Any],
) -> dict[str, Any]:
    """Worker function to chunk a single canonical JSON file.

    Must be at module level to be picklable by ProcessPoolExecutor.

    Args:
        json_path: Path to the canonical JSON file.
        output_path: Path to write the chunked output.
        settings_dict: Serialized settings for chunker configuration.

    Returns:
        Result dict with status and stats.
    """
    try:
        # Load canonical document
        with open(json_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        pmcid = json_path.stem
        doc_metadata = doc_data.get("metadata", {})

        # Initialize chunkers with settings
        semantic_chunker = SemanticChunker(
            target_tokens=settings_dict.get("chunk_target_tokens", 512),
            max_tokens=settings_dict.get("chunk_max_tokens", 600),
            min_tokens=settings_dict.get("chunk_min_tokens", 50),
            overlap_tokens=settings_dict.get("chunk_overlap_tokens", 75),
        )
        table_chunker = TableChunker(
            max_tokens=settings_dict.get("chunk_max_tokens", 600),
        )
        validator = ChunkValidator(
            min_tokens=settings_dict.get("chunk_min_tokens", 50),
        )

        all_chunks = []

        # 1. Abstract
        abstract_text = doc_data.get("abstract", "")
        abstract_chunks = semantic_chunker.chunk_abstract(
            abstract_text, pmcid, doc_metadata
        )
        all_chunks.extend(abstract_chunks)

        # 2. Sections
        sections = doc_data.get("sections", [])
        section_chunks = semantic_chunker.chunk_sections(
            sections, pmcid, doc_metadata
        )
        all_chunks.extend(section_chunks)

        # 3. Tables
        tables = doc_data.get("tables", [])
        table_chunks = table_chunker.chunk_tables(
            tables, pmcid, doc_metadata
        )
        all_chunks.extend(table_chunks)

        # 4. Figures
        figures = doc_data.get("figures", [])
        figure_chunks = table_chunker.chunk_figures(
            figures, pmcid, doc_metadata
        )
        all_chunks.extend(figure_chunks)

        # 5. Validate
        valid_chunks, rejected, report = validator.validate(all_chunks)

        # 6. Build output
        chunked_doc = ChunkedDocument(
            document_metadata=doc_metadata,
            chunks=valid_chunks,
        )

        # 7. Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked_doc.to_dict(), f, indent=2, ensure_ascii=False)

        # Compute per-document stats
        token_counts = [c.token_count for c in valid_chunks]

        return {
            "pmcid": pmcid,
            "status": "success",
            "total_chunks": len(valid_chunks),
            "rejected": len(rejected),
            "text_chunks": sum(1 for c in valid_chunks if c.chunk_type == "TEXT"),
            "table_chunks": sum(1 for c in valid_chunks if c.chunk_type == "TABLE"),
            "figure_chunks": sum(
                1 for c in valid_chunks if c.chunk_type == "FIGURE_CAPTION"
            ),
            "abstract_chunks": sum(
                1 for c in valid_chunks if c.chunk_type == "ABSTRACT"
            ),
            "largest": max(token_counts) if token_counts else 0,
            "smallest": min(token_counts) if token_counts else 0,
            "total_tokens": sum(token_counts),
        }

    except Exception as e:
        return {"pmcid": json_path.stem, "status": "failed", "error": str(e)}


class ChunkPipeline:
    """Orchestrates batch chunking with multiprocessing and resume support.

    Mirrors the ParserPipeline pattern from Phase 2 for consistency.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parsed_json_dir = settings.parsed_json_dir
        self.chunks_dir = settings.chunks_dir
        self.chunking_logs_dir = settings.chunking_logs_dir

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.chunking_logs_dir.mkdir(parents=True, exist_ok=True)

        self.stats = ChunkStats()
        self.stats_file = self.chunking_logs_dir / "chunk_statistics.json"

        # Determine worker count
        self.workers = settings.max_chunk_workers or max(
            1, multiprocessing.cpu_count() - 1
        )

        self._interrupted = False

        # Serialize settings for worker processes
        self._settings_dict = {
            "chunk_target_tokens": settings.chunk_target_tokens,
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_min_tokens": settings.chunk_min_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        }

    def _setup_logging(self, verbose: bool) -> None:
        """Setup logging for the chunking pipeline."""
        chunk_logger = logging.getLogger("chunking")
        chunk_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not chunk_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # General log
            fh = logging.FileHandler(
                self.chunking_logs_dir / "chunk.log", encoding="utf-8"
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            chunk_logger.addHandler(fh)

            # Error log
            eh = logging.FileHandler(
                self.chunking_logs_dir / "chunk_errors.log", encoding="utf-8"
            )
            eh.setLevel(logging.WARNING)
            eh.setFormatter(formatter)
            chunk_logger.addHandler(eh)

            # Console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if verbose else logging.INFO)
            ch.setFormatter(formatter)
            chunk_logger.addHandler(ch)

    def _save_stats(self) -> None:
        """Save aggregate statistics to JSON."""
        self.stats_file.write_text(
            json.dumps(asdict(self.stats), indent=2), encoding="utf-8"
        )

    def run(
        self,
        limit: int | None = None,
        resume: bool = True,
        force: bool = False,
        verbose: bool = False,
        target_pmcid: str | None = None,
    ) -> ChunkStats:
        """Execute the chunking pipeline over canonical JSON files.

        Args:
            limit: Maximum number of files to process.
            resume: Whether to skip already-chunked files.
            force: If True, overwrite existing chunk files.
            verbose: Enable DEBUG-level logging.
            target_pmcid: Process only this specific PMCID.

        Returns:
            ChunkStats with aggregate results.
        """
        self._setup_logging(verbose)
        chunk_logger = logging.getLogger("chunking")

        if not resume or force:
            self.stats = ChunkStats()

        # Find files to process
        json_files: list[Path] = []
        if target_pmcid:
            target_path = self.parsed_json_dir / f"{target_pmcid}.json"
            if target_path.exists():
                json_files.append(target_path)
            else:
                chunk_logger.error(
                    "Target PMCID %s not found in %s",
                    target_pmcid, self.parsed_json_dir,
                )
                return self.stats
        else:
            json_files = sorted(self.parsed_json_dir.glob("*.json"))
            if limit:
                json_files = json_files[:limit]

        if not json_files:
            chunk_logger.info("No canonical JSON files found to chunk.")
            return self.stats

        chunk_logger.info(
            "Found %d canonical JSON files. Using %d workers.",
            len(json_files), self.workers,
        )

        # Build task list with resume/force logic
        tasks: list[tuple[Path, Path]] = []
        for json_path in json_files:
            output_path = self.chunks_dir / f"{json_path.stem}_chunks.json"
            if not force and resume and output_path.exists():
                self.stats.skipped += 1
                continue
            tasks.append((json_path, output_path))

        if not tasks:
            chunk_logger.info(
                "All files already chunked (use --force to re-chunk)."
            )
            return self.stats

        chunk_logger.info("Processing %d files...", len(tasks))

        # Setup graceful interruption
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Accumulators for computing averages
        all_token_counts: list[int] = []
        global_largest = 0
        global_smallest = float("inf")

        try:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(
                        _chunk_worker, json_path, output_path, self._settings_dict
                    ): json_path
                    for json_path, output_path in tasks
                }

                with tqdm(
                    total=len(futures), desc="Chunking documents", unit="doc"
                ) as pbar:
                    for future in as_completed(futures):
                        if self._interrupted:
                            chunk_logger.info(
                                "Interrupted. Cancelling remaining tasks..."
                            )
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                        try:
                            result = future.result()
                            status = result["status"]
                            pmcid = result["pmcid"]

                            if status == "success":
                                self.stats.total_documents += 1
                                self.stats.total_chunks += result["total_chunks"]
                                self.stats.text_chunks += result["text_chunks"]
                                self.stats.table_chunks += result["table_chunks"]
                                self.stats.figure_chunks += result["figure_chunks"]
                                self.stats.abstract_chunks += result[
                                    "abstract_chunks"
                                ]
                                self.stats.rejected_chunks += result["rejected"]

                                if result["largest"] > global_largest:
                                    global_largest = result["largest"]
                                if (
                                    result["smallest"] < global_smallest
                                    and result["smallest"] > 0
                                ):
                                    global_smallest = result["smallest"]

                                total_tokens = result.get("total_tokens", 0)
                                if total_tokens > 0:
                                    all_token_counts.append(total_tokens)

                                chunk_logger.debug(
                                    "Chunked %s → %d chunks",
                                    pmcid, result["total_chunks"],
                                )
                            else:
                                self.stats.failed += 1
                                err = result.get("error", "Unknown error")
                                chunk_logger.error(
                                    "Failed to chunk %s: %s", pmcid, err
                                )

                        except Exception as e:
                            self.stats.failed += 1
                            chunk_logger.error("Worker exception: %s", e)

                        pbar.set_postfix(
                            docs=self.stats.total_documents,
                            chunks=self.stats.total_chunks,
                            failed=self.stats.failed,
                        )
                        pbar.update(1)

        finally:
            signal.signal(signal.SIGINT, original_sigint)

            # Compute final averages
            self.stats.largest_chunk = global_largest
            self.stats.smallest_chunk = (
                int(global_smallest) if global_smallest != float("inf") else 0
            )
            if self.stats.total_chunks > 0 and all_token_counts:
                self.stats.avg_chunk_size = round(
                    sum(all_token_counts) / self.stats.total_chunks, 2
                )

            self._save_stats()
            chunk_logger.info("Chunking complete.")
            chunk_logger.info("Stats: %s", asdict(self.stats))

        return self.stats

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT for graceful shutdown."""
        if self._interrupted:
            sys.exit(1)
        print(
            "\nInterrupt received. Finishing active tasks... "
            "Press Ctrl+C again to force exit."
        )
        self._interrupted = True

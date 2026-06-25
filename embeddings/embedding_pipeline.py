"""
Healthcare Knowledge Navigator — Embedding Pipeline.

Top-level orchestrator for the embedding generation pipeline.
Mirrors ``ChunkPipeline`` from Phase 3 for consistency.

Responsibilities:
    - Device detection and model initialization
    - Chunk file discovery with resume/force logic
    - Per-document embedding generation and validation
    - Statistics collection and logging
    - Graceful SIGINT handling
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from configs.settings import Settings
from embeddings.batch_processor import cleanup_gpu_memory
from embeddings.device_manager import detect_device
from embeddings.embedding_generator import EmbeddingGenerator
from embeddings.embedding_model import MedCPTEmbeddingModel
from embeddings.embedding_statistics import EmbeddingStats, StatisticsCollector
from embeddings.embedding_validator import EmbeddingValidator

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Orchestrates batch embedding generation with resume support.

    Mirrors the ``ChunkPipeline`` pattern from Phase 3 for consistency:
    file discovery → resume filtering → per-document processing →
    statistics → graceful shutdown.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the embedding pipeline.

        Args:
            settings: Application settings with path and model configuration.
        """
        self.settings = settings
        self.chunks_dir = settings.chunks_dir
        self.embeddings_dir = settings.embeddings_dir
        self.embedding_logs_dir = settings.embedding_logs_dir

        # Ensure output directories exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_logs_dir.mkdir(parents=True, exist_ok=True)

        self._interrupted = False

    def _setup_logging(self, verbose: bool) -> None:
        """Configure logging for the embedding pipeline.

        Args:
            verbose: If True, enable DEBUG-level logging.
        """
        emb_logger = logging.getLogger("embedding")
        emb_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not emb_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # General log
            fh = logging.FileHandler(
                self.embedding_logs_dir / "embedding.log",
                encoding="utf-8",
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            emb_logger.addHandler(fh)

            # Error log
            eh = logging.FileHandler(
                self.embedding_logs_dir / "embedding_errors.log",
                encoding="utf-8",
            )
            eh.setLevel(logging.WARNING)
            eh.setFormatter(formatter)
            emb_logger.addHandler(eh)

            # Console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if verbose else logging.INFO)
            ch.setFormatter(formatter)
            emb_logger.addHandler(ch)

    def run(
        self,
        limit: int | None = None,
        resume: bool = True,
        force: bool = False,
        verbose: bool = False,
        target_pmcid: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> EmbeddingStats:
        """Execute the embedding generation pipeline.

        Args:
            limit: Maximum number of documents to process.
            resume: Skip documents with existing embeddings.
            force: Re-generate embeddings even if output exists.
            verbose: Enable DEBUG-level logging.
            target_pmcid: Process only this specific PMCID.
            batch_size: Chunks per GPU batch (overrides settings).
            device: Force a specific device (cuda/mps/cpu).

        Returns:
            EmbeddingStats with aggregate results.
        """
        self._setup_logging(verbose)
        emb_logger = logging.getLogger("embedding")

        # Resolve batch size
        effective_batch_size = batch_size or self.settings.embedding_batch_size

        # 1. Detect device
        device_info = detect_device(preferred=device)
        emb_logger.info("Device: %s (%s)", device_info.device_name, device_info.device_type)

        # 2. Initialize model
        emb_logger.info("Loading embedding model: %s", self.settings.embedding_model_name)
        model = MedCPTEmbeddingModel(
            model_name_or_path=self.settings.embedding_model_name,
            device_info=device_info,
        )
        emb_logger.info(
            "Model loaded — dimension: %d", model.embedding_dimension
        )

        # 3. Initialize components
        generator = EmbeddingGenerator(model=model, batch_size=effective_batch_size)
        validator = EmbeddingValidator(expected_dimension=model.embedding_dimension)
        collector = StatisticsCollector(
            model=model.model_name,
            device=device_info.device_type,
            dimension=model.embedding_dimension,
        )

        # 4. Discover chunk files
        chunk_files = self._discover_files(target_pmcid, limit)
        if not chunk_files:
            emb_logger.info("No chunk files found to embed.")
            return collector.finalize()

        emb_logger.info("Found %d chunk files.", len(chunk_files))

        # 5. Filter for resume/force
        tasks = self._filter_tasks(chunk_files, resume, force, collector)
        if not tasks:
            emb_logger.info(
                "All documents already embedded (use --force to re-embed)."
            )
            return collector.finalize()

        emb_logger.info("Processing %d documents (batch_size=%d)...", len(tasks), effective_batch_size)

        # 6. Setup graceful interruption
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        collector.start()

        try:
            with tqdm(
                total=len(tasks),
                desc="Embedding documents",
                unit="doc",
            ) as pbar:
                for chunk_file, output_file in tasks:
                    if self._interrupted:
                        emb_logger.info("Interrupted. Stopping...")
                        break

                    try:
                        self._process_document(
                            chunk_file=chunk_file,
                            output_file=output_file,
                            generator=generator,
                            validator=validator,
                            collector=collector,
                            emb_logger=emb_logger,
                        )
                    except Exception as e:
                        collector.record_failure()
                        emb_logger.error(
                            "Failed to embed %s: %s",
                            chunk_file.stem,
                            e,
                        )

                    pbar.update(1)
                    pbar.set_postfix(
                        docs=collector._total_documents,
                        chunks=collector._total_chunks,
                        failed=collector._documents_failed,
                    )

        finally:
            signal.signal(signal.SIGINT, original_sigint)

            # Save statistics
            stats = collector.finalize()
            stats_path = self.embedding_logs_dir / "embedding_statistics.json"
            collector.save(stats_path)

            emb_logger.info("Embedding generation complete.")
            emb_logger.info("Stats: %s", asdict(stats))

        return stats

    def _discover_files(
        self,
        target_pmcid: str | None,
        limit: int | None,
    ) -> list[Path]:
        """Discover chunk files to process.

        Args:
            target_pmcid: If set, only find this specific document.
            limit: Maximum number of files to return.

        Returns:
            Sorted list of chunk file paths.
        """
        if target_pmcid:
            # Support both "PMC123456" and "PMC123456_chunks"
            candidates = [
                self.chunks_dir / f"{target_pmcid}_chunks.json",
                self.chunks_dir / f"{target_pmcid}.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]
            logger.warning(
                "Target PMCID %s not found in %s",
                target_pmcid,
                self.chunks_dir,
            )
            return []

        files = sorted(self.chunks_dir.glob("*_chunks.json"))
        if limit:
            files = files[:limit]
        return files

    def _filter_tasks(
        self,
        chunk_files: list[Path],
        resume: bool,
        force: bool,
        collector: StatisticsCollector,
    ) -> list[tuple[Path, Path]]:
        """Filter files based on resume/force logic.

        Args:
            chunk_files: All discovered chunk files.
            resume: Whether to skip existing outputs.
            force: Whether to overwrite existing outputs.
            collector: Statistics collector for recording skips.

        Returns:
            List of (chunk_file, output_file) tuples to process.
        """
        tasks: list[tuple[Path, Path]] = []

        for chunk_file in chunk_files:
            # Derive output path: PMC123456_chunks.json → PMC123456_embeddings.json
            stem = chunk_file.stem.replace("_chunks", "")
            output_file = self.embeddings_dir / f"{stem}_embeddings.json"

            if not force and resume and output_file.exists():
                collector.record_skip()
                continue

            tasks.append((chunk_file, output_file))

        return tasks

    def _process_document(
        self,
        chunk_file: Path,
        output_file: Path,
        generator: EmbeddingGenerator,
        validator: EmbeddingValidator,
        collector: StatisticsCollector,
        emb_logger: logging.Logger,
    ) -> None:
        """Process a single document: embed → validate → write.

        Args:
            chunk_file: Input chunk JSON path.
            output_file: Output embedding JSON path.
            generator: The embedding generator.
            validator: The embedding validator.
            collector: Statistics collector.
            emb_logger: Logger for embedding events.
        """
        pmcid = chunk_file.stem.replace("_chunks", "")

        # Generate embeddings
        collector.start_batch()
        embedded_doc = generator.generate(chunk_file)
        chunk_count = len(embedded_doc.chunks)
        collector.end_batch(chunk_count)

        # Record norms
        norms = generator.compute_norms(embedded_doc.chunks)
        if norms:
            collector.record_norms(norms)

        # Validate embeddings
        doc_dict = embedded_doc.to_dict()
        valid_chunks, report = validator.validate_document(doc_dict["chunks"])

        if report.total_invalid > 0:
            emb_logger.warning(
                "%s: %d/%d embeddings invalid — excluded from output",
                pmcid,
                report.total_invalid,
                report.total_checked,
            )

        # Update output with only valid chunks
        doc_dict["chunks"] = valid_chunks

        # Write to disk
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)

        collector.record_document(chunk_count=len(valid_chunks))

        emb_logger.debug(
            "Embedded %s → %d chunks (%d valid)",
            pmcid,
            chunk_count,
            len(valid_chunks),
        )

        # Free GPU memory after each document
        cleanup_gpu_memory()

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT for graceful shutdown."""
        if self._interrupted:
            sys.exit(1)
        print(
            "\nInterrupt received. Finishing current document... "
            "Press Ctrl+C again to force exit."
        )
        self._interrupted = True

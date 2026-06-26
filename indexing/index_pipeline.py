"""
Healthcare Knowledge Navigator — Index Pipeline.

Top-level orchestrator for the Qdrant indexing pipeline.  Mirrors the
``EmbeddingPipeline`` pattern from Phase 4 for consistency.

Responsibilities:
    - Qdrant connection and health check
    - Collection creation / recreation
    - Embedding file discovery with resume logic
    - Per-document indexing via IndexBuilder
    - Post-indexing validation via IndexValidator
    - Statistics collection and persistence
    - Graceful SIGINT handling
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from configs.settings import Settings
from indexing.collection_manager import CollectionConfig, CollectionManager
from indexing.index_builder import IndexBuilder
from indexing.index_statistics import IndexStats, IndexStatsCollector
from indexing.index_validator import IndexValidator
from indexing.qdrant_manager import QdrantManager
from indexing.sparse_index import SparseEncoder

logger = logging.getLogger(__name__)


class IndexPipeline:
    """Orchestrates batch indexing of embeddings into Qdrant.

    Pipeline flow:
        1. Connect to Qdrant (health check)
        2. Create / recreate collection
        3. Discover embedding files
        4. For each file: build points → filter existing → batch upsert
        5. Run collection-level validation
        6. Save statistics
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the index pipeline.

        Args:
            settings: Application settings with Qdrant and path config.
        """
        self.settings = settings
        self.embeddings_dir = settings.embeddings_dir
        self.indexing_logs_dir = settings.indexing_logs_dir
        self.indexing_logs_dir.mkdir(parents=True, exist_ok=True)
        self._interrupted = False

    def _setup_logging(self, verbose: bool) -> None:
        """Configure logging for the indexing pipeline.

        Args:
            verbose: If True, enable DEBUG-level logging.
        """
        idx_logger = logging.getLogger("indexing")
        idx_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not idx_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # General log
            fh = logging.FileHandler(
                self.indexing_logs_dir / "index.log",
                encoding="utf-8",
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            idx_logger.addHandler(fh)

            # Error log
            eh = logging.FileHandler(
                self.indexing_logs_dir / "index_errors.log",
                encoding="utf-8",
            )
            eh.setLevel(logging.WARNING)
            eh.setFormatter(formatter)
            idx_logger.addHandler(eh)

            # Console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if verbose else logging.INFO)
            ch.setFormatter(formatter)
            idx_logger.addHandler(ch)

    def run(
        self,
        limit: int | None = None,
        resume: bool = True,
        force: bool = False,
        recreate: bool = False,
        batch_size: int | None = None,
        verbose: bool = False,
        target_pmcid: str | None = None,
    ) -> IndexStats:
        """Execute the indexing pipeline.

        Args:
            limit: Maximum number of embedding files to process.
            resume: Skip fully-indexed documents.
            force: Re-index all chunks even if they exist.
            recreate: Drop and recreate the collection before indexing.
            batch_size: Points per upsert batch (overrides settings).
            verbose: Enable DEBUG-level logging.
            target_pmcid: Process only this specific PMCID.

        Returns:
            IndexStats with aggregate results.
        """
        self._setup_logging(verbose)
        idx_logger = logging.getLogger("indexing")
        effective_batch_size = batch_size or self.settings.indexing_batch_size

        # 1. Connect to Qdrant
        idx_logger.info(
            "Connecting to Qdrant at %s:%d...",
            self.settings.qdrant_host,
            self.settings.qdrant_port,
        )
        manager = QdrantManager(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )

        if not manager.health_check():
            idx_logger.error("Qdrant is not reachable. Aborting.")
            raise ConnectionError(
                f"Cannot connect to Qdrant at "
                f"{self.settings.qdrant_host}:{self.settings.qdrant_port}"
            )

        idx_logger.info("Qdrant connection OK.")

        collector = IndexStatsCollector()

        try:
            client = manager.client

            # 2. Collection setup
            config = CollectionConfig(
                collection_name=self.settings.collection_name,
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                dense_vector_size=self.settings.dense_vector_size,
            )
            col_manager = CollectionManager(client, config)

            if recreate:
                col_manager.recreate_collection()
            else:
                col_manager.create_collection()

            # 3. Initialize components
            idx_logger.info("Loading sparse BM25 encoder...")
            sparse_encoder = SparseEncoder(model_name=self.settings.sparse_model_name)

            builder = IndexBuilder(
                client=client,
                collection_name=self.settings.collection_name,
                sparse_encoder=sparse_encoder,
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                batch_size=effective_batch_size,
            )

            # 4. Discover embedding files
            emb_files = self._discover_files(target_pmcid, limit)
            if not emb_files:
                idx_logger.info("No embedding files found to index.")
                return collector.finalize()

            idx_logger.info(
                "Found %d embedding file(s). batch_size=%d, force=%s",
                len(emb_files),
                effective_batch_size,
                force,
            )

            # 5. Setup graceful interruption
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_interrupt)

            collector.start()

            try:
                with tqdm(
                    total=len(emb_files),
                    desc="Indexing documents",
                    unit="doc",
                ) as pbar:
                    for emb_file in emb_files:
                        if self._interrupted:
                            idx_logger.info("Interrupted. Stopping gracefully...")
                            break

                        try:
                            result = builder.index_file(emb_file, force=force)
                            if result.chunks_indexed > 0 or result.chunks_skipped > 0:
                                collector.record_document(
                                    chunks_indexed=result.chunks_indexed,
                                    chunks_skipped=result.chunks_skipped,
                                )
                            elif result.errors == 0:
                                collector.record_skip()

                            if result.errors > 0:
                                collector.record_failure()

                        except Exception as e:
                            collector.record_failure()
                            idx_logger.error(
                                "Failed to index %s: %s",
                                emb_file.stem,
                                e,
                            )

                        pbar.update(1)
                        pbar.set_postfix(
                            docs=collector._documents_indexed,
                            chunks=collector._chunks_indexed,
                            skip=collector._chunks_skipped,
                        )

            finally:
                signal.signal(signal.SIGINT, original_sigint)

            # 6. Validate collection
            idx_logger.info("Running post-indexing validation...")
            validator = IndexValidator(
                client=client,
                collection_name=self.settings.collection_name,
                dense_vector_name=self.settings.dense_vector_name,
                expected_dimension=self.settings.dense_vector_size,
            )
            report = validator.validate_collection()
            for msg in report.messages:
                idx_logger.info("Validation: %s", msg)

            # 7. Save statistics
            collection_size = report.total_points
            stats = collector.finalize(collection_size)
            stats_path = self.indexing_logs_dir / "index_statistics.json"
            collector.save(stats_path, collection_size)

            idx_logger.info("Indexing pipeline complete.")
            idx_logger.info("Stats: %s", asdict(stats))

            return stats

        finally:
            manager.close()

    def _discover_files(
        self,
        target_pmcid: str | None,
        limit: int | None,
    ) -> list[Path]:
        """Discover embedding files to process.

        Args:
            target_pmcid: If set, only find this specific document.
            limit: Maximum number of files to return.

        Returns:
            Sorted list of embedding file paths.
        """
        if target_pmcid:
            candidates = [
                self.embeddings_dir / f"{target_pmcid}_embeddings.json",
                self.embeddings_dir / f"{target_pmcid}.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]
            logger.warning(
                "Target PMCID %s not found in %s",
                target_pmcid,
                self.embeddings_dir,
            )
            return []

        files = sorted(self.embeddings_dir.glob("*_embeddings.json"))
        if limit:
            files = files[:limit]
        return files

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT for graceful shutdown."""
        if self._interrupted:
            sys.exit(1)
        print(
            "\nInterrupt received. Finishing current document... "
            "Press Ctrl+C again to force exit."
        )
        self._interrupted = True

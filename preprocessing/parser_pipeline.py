"""
Healthcare Knowledge Navigator — Parser Pipeline.

Orchestrates multiprocessing for batch parsing of PMC JATS XML files.
Provides structured logging for parsing success, failures, and statistics.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from configs.settings import Settings
from preprocessing.xml_parser import XMLParser

logger = logging.getLogger(__name__)


@dataclass
class ParseStats:
    """Aggregate statistics for a parsing run."""

    parsed: int = 0
    skipped: int = 0
    failed: int = 0
    invalid: int = 0


# Top-level worker function for multiprocessing
def _parse_worker(xml_path: Path, output_path: Path, invalid_dir: Path) -> dict[str, Any]:
    """
    Worker function to parse a single XML file.
    Must be at module level to be picklable by ProcessPoolExecutor.
    """
    # Create parser locally in worker
    parser = XMLParser(invalid_json_dir=invalid_dir)
    
    try:
        success = parser.parse_file(xml_path, output_path)
        if success:
            return {"pmcid": xml_path.stem, "status": "success"}
        else:
            return {"pmcid": xml_path.stem, "status": "invalid"}
    except Exception as e:
        return {"pmcid": xml_path.stem, "status": "failed", "error": str(e)}


class ParserPipeline:
    """
    Orchestrates batch parsing of XML files with multiprocessing and resume support.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_xml_dir = settings.raw_xml_dir
        self.parsed_json_dir = settings.parsed_json_dir
        self.parsing_logs_dir = settings.parsing_logs_dir
        self.invalid_json_dir = settings.invalid_json_dir

        self.parsing_logs_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_json_dir.mkdir(parents=True, exist_ok=True)
        self.invalid_json_dir.mkdir(parents=True, exist_ok=True)

        self.stats = ParseStats()
        self.stats_file = self.parsing_logs_dir / "parse_statistics.json"

        # Determine worker count
        self.workers = settings.max_parse_workers or max(1, multiprocessing.cpu_count() - 1)

        self._interrupted = False

    def _setup_logging(self, verbose: bool) -> None:
        """Setup logging for the parsing pipeline."""
        parse_logger = logging.getLogger("parsing")
        parse_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not parse_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            
            # General log
            fh = logging.FileHandler(self.parsing_logs_dir / "parse.log", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            parse_logger.addHandler(fh)

            # Error log
            eh = logging.FileHandler(self.parsing_logs_dir / "parse_errors.log", encoding="utf-8")
            eh.setLevel(logging.WARNING)
            eh.setFormatter(formatter)
            parse_logger.addHandler(eh)

            # Console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if verbose else logging.INFO)
            ch.setFormatter(formatter)
            parse_logger.addHandler(ch)

    def _load_stats(self) -> None:
        if self.stats_file.exists():
            try:
                data = json.loads(self.stats_file.read_text(encoding="utf-8"))
                self.stats = ParseStats(**data)
            except Exception:
                pass

    def _save_stats(self) -> None:
        self.stats_file.write_text(json.dumps(asdict(self.stats), indent=2), encoding="utf-8")

    def run(self, limit: int | None = None, resume: bool = True, verbose: bool = False, target_pmcid: str | None = None) -> ParseStats:
        """
        Execute the parsing pipeline over raw XML files.
        """
        self._setup_logging(verbose)
        parse_logger = logging.getLogger("parsing")
        
        if resume:
            self._load_stats()
        else:
            self.stats = ParseStats()

        # Find files to process
        xml_files = []
        if target_pmcid:
            target_path = self.raw_xml_dir / f"{target_pmcid}.xml"
            if target_path.exists():
                xml_files.append(target_path)
            else:
                parse_logger.error(f"Target PMCID {target_pmcid} not found in {self.raw_xml_dir}")
                return self.stats
        else:
            xml_files = list(self.raw_xml_dir.glob("*.xml"))
            if limit:
                xml_files = xml_files[:limit]

        if not xml_files:
            parse_logger.info("No XML files found to parse.")
            return self.stats

        parse_logger.info(f"Found {len(xml_files)} XML files to parse. Using {self.workers} workers.")

        tasks = []
        for xml_path in xml_files:
            output_path = self.parsed_json_dir / f"{xml_path.stem}.json"
            if resume and output_path.exists():
                self.stats.skipped += 1
                continue
            tasks.append((xml_path, output_path, self.invalid_json_dir))

        if not tasks:
            parse_logger.info("All files already parsed (use --no-resume to reparse).")
            return self.stats

        parse_logger.info(f"Processing {len(tasks)} files...")

        # Setup graceful interruption
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(_parse_worker, xml_path, output_path, invalid_dir): xml_path
                    for xml_path, output_path, invalid_dir in tasks
                }

                with tqdm(total=len(futures), desc="Parsing XML", unit="file") as pbar:
                    for future in as_completed(futures):
                        if self._interrupted:
                            parse_logger.info("Interrupted. Cancelling remaining tasks...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        
                        try:
                            result = future.result()
                            status = result["status"]
                            pmcid = result["pmcid"]
                            
                            if status == "success":
                                self.stats.parsed += 1
                                parse_logger.debug(f"Successfully parsed {pmcid}")
                            elif status == "invalid":
                                self.stats.invalid += 1
                                parse_logger.warning(f"Invalid document quarantined: {pmcid}")
                            else:
                                self.stats.failed += 1
                                err = result.get("error", "Unknown error")
                                parse_logger.error(f"Failed to parse {pmcid}: {err}")
                                
                        except Exception as e:
                            self.stats.failed += 1
                            parse_logger.error(f"Worker exception: {e}")

                        pbar.set_postfix(parsed=self.stats.parsed, failed=self.stats.failed, invalid=self.stats.invalid)
                        pbar.update(1)

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self._save_stats()
            parse_logger.info("Parsing complete.")
            parse_logger.info(f"Stats: {asdict(self.stats)}")

        return self.stats

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        if self._interrupted:
            sys.exit(1)
        print("\nInterrupt received. Finishing active tasks... Press Ctrl+C again to force exit.")
        self._interrupted = True

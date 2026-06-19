"""
Healthcare Knowledge Navigator — Environment Verification Script.

Validates that all required dependencies, services, and configuration
are correctly set up before running the application.
"""

import importlib
import os
import sys
from pathlib import Path

# ── Resolve project root ─────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ── Required Python Packages ─────────────────────────────────────────
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("streamlit", "Streamlit"),
    ("pydantic", "Pydantic"),
    ("pydantic_settings", "Pydantic Settings"),
    ("groq", "Groq SDK"),
    ("qdrant_client", "Qdrant Client"),
    ("sentence_transformers", "Sentence Transformers"),
    ("transformers", "Transformers"),
    ("torch", "PyTorch"),
    ("lxml", "lxml"),
    ("bs4", "BeautifulSoup4"),
    ("rank_bm25", "Rank BM25"),
    ("dotenv", "python-dotenv"),
    ("httpx", "HTTPX"),
    ("rich", "Rich"),
]

# ── Required Environment Variables ───────────────────────────────────
REQUIRED_ENV_VARS: list[str] = [
    "GROQ_API_KEY",
]

OPTIONAL_ENV_VARS: list[str] = [
    "QDRANT_HOST",
    "QDRANT_PORT",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "COLLECTION_NAME",
    "TOP_K",
    "FINAL_K",
    "LOG_LEVEL",
]

# ── Required Directories ─────────────────────────────────────────────
REQUIRED_DIRS: list[Path] = [
    PROJECT_ROOT / "data" / "raw_xml",
    PROJECT_ROOT / "data" / "parsed_json",
    PROJECT_ROOT / "data" / "chunks",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "models",
]


def check_python_version() -> bool:
    """Verify Python version is 3.12+."""
    print("\n🐍 Python Version Check")
    print("-" * 40)
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info.micro}"

    if major >= 3 and minor >= 12:
        print(f"   ✅ Python {version_str}")
        return True
    else:
        print(f"   ❌ Python {version_str} — requires 3.12+")
        return False


def check_packages() -> tuple[int, int]:
    """Check that all required Python packages are importable.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    print("\n📦 Python Package Check")
    print("-" * 40)
    passed = 0
    failed = 0

    for import_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"   ✅ {display_name:<25s} v{version}")
            passed += 1
        except ImportError:
            print(f"   ❌ {display_name:<25s} — NOT INSTALLED")
            failed += 1

    return passed, failed


def check_env_vars() -> tuple[int, int]:
    """Check that required environment variables are set.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    # Load .env file if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    print("\n🔐 Environment Variables Check")
    print("-" * 40)

    passed = 0
    failed = 0

    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"   ✅ {var:<25s} = {masked}")
            passed += 1
        else:
            print(f"   ❌ {var:<25s} — NOT SET")
            failed += 1

    print()
    for var in OPTIONAL_ENV_VARS:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var:<25s} = {value}")
        else:
            print(f"   ⚠️  {var:<25s} — not set (using default)")

    return passed, failed


def check_directories() -> tuple[int, int]:
    """Check that required directories exist.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    print("\n📁 Directory Check")
    print("-" * 40)

    passed = 0
    failed = 0

    for directory in REQUIRED_DIRS:
        relative = directory.relative_to(PROJECT_ROOT)
        if directory.exists():
            print(f"   ✅ {relative}/")
            passed += 1
        else:
            print(f"   ❌ {relative}/ — MISSING")
            failed += 1

    return passed, failed


def check_docker_services() -> None:
    """Check connectivity to Docker services (best-effort)."""
    print("\n🐳 Docker Services Check")
    print("-" * 40)

    # Check Qdrant
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    try:
        import httpx
        response = httpx.get(
            f"http://{qdrant_host}:{qdrant_port}/healthz",
            timeout=3.0,
        )
        if response.status_code == 200:
            print(f"   ✅ Qdrant ({qdrant_host}:{qdrant_port}) — healthy")
        else:
            print(f"   ⚠️  Qdrant ({qdrant_host}:{qdrant_port}) — status {response.status_code}")
    except Exception:
        print(f"   ⚠️  Qdrant ({qdrant_host}:{qdrant_port}) — not reachable (start with `docker compose up -d`)")


def main() -> None:
    """Run all environment checks and print a summary."""
    print("=" * 60)
    print("  🏥 Healthcare Knowledge Navigator — Environment Check")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    # Python version
    if check_python_version():
        total_passed += 1
    else:
        total_failed += 1

    # Packages
    pkg_passed, pkg_failed = check_packages()
    total_passed += pkg_passed
    total_failed += pkg_failed

    # Environment variables
    env_passed, env_failed = check_env_vars()
    total_passed += env_passed
    total_failed += env_failed

    # Directories
    dir_passed, dir_failed = check_directories()
    total_passed += dir_passed
    total_failed += dir_failed

    # Docker (informational only)
    check_docker_services()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 Summary")
    print("=" * 60)
    print(f"\n   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {total_failed}")

    if total_failed == 0:
        print("\n   🎉 All checks passed! Environment is ready.")
    else:
        print(f"\n   ⚠️  {total_failed} check(s) failed. Review above for details.")
        print("   Run `python scripts/setup.py` to fix directory issues.")
        print("   Run `pip install -r requirements.txt` to install packages.")

    print()


if __name__ == "__main__":
    main()

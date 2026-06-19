"""
Healthcare Knowledge Navigator — Project Setup Script.

Creates the required directory structure, copies .env.example to .env
if it doesn't exist, and validates that the project is ready for
development.
"""

import shutil
import sys
from pathlib import Path

# ── Resolve project root ─────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Required Directories ─────────────────────────────────────────────
REQUIRED_DIRS: list[Path] = [
    PROJECT_ROOT / "data" / "raw_xml",
    PROJECT_ROOT / "data" / "parsed_json",
    PROJECT_ROOT / "data" / "chunks",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "docs",
]


def create_directories() -> None:
    """Create all required project directories if they don't exist."""
    print("\n📁 Creating project directories...\n")
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        relative = directory.relative_to(PROJECT_ROOT)
        print(f"   ✅ {relative}/")


def create_env_file() -> None:
    """Copy .env.example to .env if .env does not exist."""
    env_file: Path = PROJECT_ROOT / ".env"
    env_example: Path = PROJECT_ROOT / ".env.example"

    print("\n🔐 Setting up environment file...\n")

    if env_file.exists():
        print("   ⚠️  .env already exists — skipping.")
        return

    if not env_example.exists():
        print("   ❌ .env.example not found! Cannot create .env.")
        return

    shutil.copy2(env_example, env_file)
    print("   ✅ Created .env from .env.example")
    print("   📝 Remember to update .env with your actual credentials.")


def create_gitkeep_files() -> None:
    """Add .gitkeep files to empty directories so Git tracks them."""
    print("\n📌 Adding .gitkeep to empty directories...\n")
    for directory in REQUIRED_DIRS:
        gitkeep: Path = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            relative = directory.relative_to(PROJECT_ROOT)
            print(f"   ✅ {relative}/.gitkeep")


def main() -> None:
    """Run the full project setup."""
    print("=" * 60)
    print("  🏥 Healthcare Knowledge Navigator — Project Setup")
    print("=" * 60)

    create_directories()
    create_env_file()
    create_gitkeep_files()

    print("\n" + "=" * 60)
    print("  ✅ Setup complete! Next steps:")
    print("=" * 60)
    print()
    print("  1. Update .env with your credentials")
    print("  2. Install dependencies:  pip install -r requirements.txt")
    print("  3. Start Qdrant:          docker compose up -d qdrant")
    print("  4. Run the backend:       uvicorn backend.main:app --reload")
    print("  5. Run the frontend:      streamlit run frontend/app.py")
    print("  6. Verify environment:    python scripts/check_environment.py")
    print()


if __name__ == "__main__":
    main()

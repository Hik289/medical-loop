"""Read local folders and model settings from environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir():
    return Path(os.environ.get("HUMANCHOICE_DATA_DIR", ROOT / "data" / "private")).expanduser().resolve()


def runs_dir():
    return Path(os.environ.get("HUMANCHOICE_RUN_DIR", ROOT / "runs")).expanduser().resolve()


def model_name():
    return os.environ.get("HUMANCHOICE_MODEL", "gpt-5.6-sol")


def make_record_client():
    """Allow record API calls only after the user has configured an approved service."""
    if os.environ.get("HUMANCHOICE_ALLOW_RECORD_API") != "1":
        raise SystemExit("Live record calls are disabled. See docs/reproduction.md.")
    if not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Configure your approved endpoint and credentials in the environment.")
    from openai import OpenAI

    return OpenAI(max_retries=6, timeout=180)

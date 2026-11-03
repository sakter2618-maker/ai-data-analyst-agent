"""AI-Powered Data Analyst Automation Agent — core package."""

from .cleaner import clean_all, clean_table  # noqa: F401
from .loader import load_bytes, load_many, load_path  # noqa: F401
from .quality import find_relationships, quality_report_md, quality_scores  # noqa: F401
from .stats import analyze  # noqa: F401

__version__ = "0.1.0"

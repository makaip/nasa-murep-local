"""
Run this file directly to execute the full pipeline.
"""

import warnings

from pickler_pipeline import run_pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    run_pipeline()

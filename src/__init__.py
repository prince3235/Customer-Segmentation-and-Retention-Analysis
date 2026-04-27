"""
src/__init__.py
===============
Makes `src` a proper Python package.
Allows clean imports like:
    from src.preprocessing import run_preprocessing
    from src.rfm           import build_rfm_table
    from src.clustering    import run_clustering
    from src.model         import run_model_pipeline
"""

from src.preprocessing import run_preprocessing, data_summary
from src.rfm           import build_rfm_table, segment_summary
from src.clustering    import run_clustering
from src.model         import run_model_pipeline, predict_churn, load_model

__all__ = [
    "run_preprocessing",
    "data_summary",
    "build_rfm_table",
    "segment_summary",
    "run_clustering",
    "run_model_pipeline",
    "predict_churn",
    "load_model",
]

__version__ = "1.0.0"
__author__  = "Customer Segmentation ML Pipeline"
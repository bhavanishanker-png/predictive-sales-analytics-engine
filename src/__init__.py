"""SaaS sales analytics pipeline modules."""

# Imports are kept lazy here to avoid loading heavy deps (torch, transformers)
# when notebooks only need a subset of src. Import directly from submodules.

__all__ = [
    "GatedFusion",
    "HybridExplainer",
    "HybridSalesPredictor",
    "SalesPredictor",
    "TextEncoder",
    "detect_text_column",
    "parse_plain_text",
    "parse_uploaded_file",
]

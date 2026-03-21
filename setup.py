"""
Dependency metadata for the project (no installable Python package; logic lives in notebooks).
`pip install -e .` still installs the listed requirements.
"""

from setuptools import setup

setup(
    name="predictive-sales-analytics-engine",
    version="1.0.0",
    description="Machine learning system for predicting SaaS sales deal outcomes using multimodal learning",
    author="Your Team Name",
    author_email="your.email@example.com",
    packages=[],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "datasets>=4.0.0",
        "jupyter>=1.0.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
        ],
        "ml": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "torch>=1.9.0",  # For deep learning if needed
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

"""
Setup configuration for LlamaAgent package.
Ready for PyPI publication.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README_COMPLETE.md").read_text(encoding="utf-8")

# Read version from _version.py
version_file = this_directory / "src" / "llamaagent" / "_version.py"
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)
version = version_dict["__version__"]

# Core dependencies
install_requires = [
    # Core
    "pydantic>=2.0.0",
    "typing-extensions>=4.5.0",
    "python-dotenv>=1.0.0",
    
    # Async
    "asyncio>=3.4.3",
    "aiohttp>=3.8.0",
    "aiofiles>=23.0.0",
    
    # CLI and UI
    "typer>=0.9.0",
    "rich>=13.0.0",
    "prompt-toolkit>=3.0.0",
    
    # Web framework
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "python-multipart>=0.0.6",
    
    # Database and storage
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.28.0",
    "redis>=5.0.0",
    
    # Data processing
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # Utilities
    "structlog>=23.0.0",
    "python-json-logger>=2.0.0",
    "tenacity>=8.2.0",
    "cachetools>=5.3.0",
]

# Optional dependencies for different providers
extras_require = {
    # LLM Providers
    "openai": ["openai>=1.0.0"],
    "anthropic": ["anthropic>=0.8.0"],
    "cohere": ["cohere>=4.0.0"],
    "together": ["together>=0.2.0"],
    "litellm": ["litellm>=1.0.0"],
    
    # Vector stores
    "chromadb": ["chromadb>=0.4.0"],
    "pinecone": ["pinecone-client>=2.2.0"],
    "weaviate": ["weaviate-client>=3.0.0"],
    
    # ML/AI tools
    "transformers": ["transformers>=4.30.0", "torch>=2.0.0"],
    "sentence-transformers": ["sentence-transformers>=2.2.0"],
    
    # Development
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    
    # All optional dependencies
    "all": [
        "openai>=1.0.0",
        "anthropic>=0.8.0", 
        "cohere>=4.0.0",
        "together>=0.2.0",
        "litellm>=1.0.0",
        "chromadb>=0.4.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
    ],
}

# Package metadata
setup(
    name="llamaagent",
    version=version,
    author="LlamaAgent Team",
    author_email="team@llamaagent.ai",
    description="A production-ready AI agent framework with multi-provider support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamaagent/llamaagent",
    project_urls={
        "Documentation": "https://llamaagent.readthedocs.io",
        "Bug Reports": "https://github.com/llamaagent/llamaagent/issues",
        "Source": "https://github.com/llamaagent/llamaagent",
        "Changelog": "https://github.com/llamaagent/llamaagent/blob/main/CHANGELOG.md",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "ai",
        "agents", 
        "llm",
        "langchain",
        "openai",
        "anthropic",
        "gpt",
        "claude",
        "machine-learning",
        "artificial-intelligence",
        "nlp",
        "natural-language-processing",
        "react-agent",
        "autonomous-agents",
        "ai-agents",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "llamaagent": [
            "py.typed",
            "data/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "llamaagent=llamaagent.cli.main:app",
            "lla=llamaagent.cli.main:app",  # Short alias
        ],
    },
    zip_safe=False,
)
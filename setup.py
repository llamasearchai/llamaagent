#!/usr/bin/env python3
"""
Setup script for LlamaAgent - Advanced AI Agent Framework

Author: Nik Jois <nikjois@llamasearch.ai>
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from _version.py
version_file = this_directory / "src" / "llamaagent" / "_version.py"
version_dict = {}
with open(version_file, encoding='utf-8') as f:
    exec(f.read(), version_dict)

# Core dependencies
install_requires = [
    # Core framework
    "pydantic>=2.0.0,<3.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "structlog>=23.0.0,<24.0.0",
    "rich>=13.0.0,<14.0.0",
    "typer>=0.9.0,<1.0.0",
    "click>=8.1.0,<9.0.0",
    
    # Async and HTTP
    "httpx[http2]>=0.25.0,<1.0.0",
    "aiofiles>=23.0.0,<24.0.0",
    "asyncio-throttle>=1.0.0,<2.0.0",
    
    # AI and LLM
    "openai>=1.0.0,<2.0.0",
    "anthropic>=0.3.0,<1.0.0",
    "litellm>=1.40.0,<2.0.0",
    
    # Web framework
    "fastapi>=0.100.0,<1.0.0",
    "uvicorn[standard]>=0.23.0,<1.0.0",
    "starlette>=0.27.0,<1.0.0",
    
    # Database and storage
    "sqlalchemy>=2.0.0,<3.0.0",
    "asyncpg>=0.28.0,<1.0.0",
    "psycopg2-binary>=2.9.0,<3.0.0",
    "redis>=4.6.0,<5.0.0",
    "aioredis>=2.0.0,<3.0.0",
    
    # Data processing
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "pyyaml>=6.0.0,<7.0.0",
    
    # Monitoring and observability
    "prometheus-client>=0.17.0,<1.0.0",
    "psutil>=5.9.0,<6.0.0",
    
    # Security
    "cryptography>=41.0.0,<43.0.0",
    "python-jose[cryptography]>=3.3.0,<4.0.0",
    "bcrypt>=4.0.0,<5.0.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0,<8.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "pytest-cov>=4.1.0,<5.0.0",
        "pytest-xdist>=3.3.0,<4.0.0",
        "pytest-mock>=3.11.0,<4.0.0",
        "coverage[toml]>=7.0.0,<8.0.0",
        "pre-commit>=3.5.0,<4.0.0",
        "mypy>=1.8.0,<2.0.0",
        "ruff>=0.1.0,<1.0.0",
        "black>=23.7.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
    ],
    "ai-extended": [
        "sentence-transformers>=2.2.0,<3.0.0",
        "chromadb>=0.4.0,<1.0.0",
    ],
    "vector": [
        "faiss-cpu>=1.7.0,<2.0.0",
        "chromadb>=0.4.0,<1.0.0",
    ],
    "distributed": [
        "celery>=5.3.0,<6.0.0",
        "redis>=4.6.0,<5.0.0",
    ],
    "monitoring": [
        "prometheus-client>=0.17.0,<1.0.0",
        "psutil>=5.9.0,<6.0.0",
    ],
    "enterprise": [
        "docker>=6.0.0,<7.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=2.0.0,<3.0.0",
        "sphinx-autodoc-typehints>=1.24.0,<2.0.0",
        "myst-parser>=2.0.0,<3.0.0",
    ],
}

# All optional dependencies
extras_require["all"] = [
    dep for deps in extras_require.values() for dep in deps
]

setup(
    name="llamaagent",
    version=version_dict["__version__"],
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Advanced AI Agent Framework with Enterprise Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikjois/llamaagent",
    project_urls={
        "Documentation": "https://nikjois.github.io/llamaagent",
        "Bug Tracker": "https://github.com/nikjois/llamaagent/issues",
        "Source Code": "https://github.com/nikjois/llamaagent",
        "Changelog": "https://github.com/nikjois/llamaagent/releases",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "llamaagent=llamaagent.cli:main",
            "llamaagent-server=llamaagent.api:run_server",
            "llamaagent-worker=llamaagent.distributed:run_worker",
        ],
        "llamaagent.tools": [
            "calculator=llamaagent.tools.calculator:CalculatorTool",
            "python_repl=llamaagent.tools.python_repl:PythonREPLTool",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ai", "agent", "llm", "automation", "enterprise", "distributed", 
        "orchestration", "tools", "reasoning", "multimodal"
    ],
    license="MIT",
    platforms=["any"],
) 
#!/usr/bin/env python
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamaagent-llamasearch",
    version="0.1.0",
    author="Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois",
    author_email="nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai",
    description="Agent framework for building autonomous LLM applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearch/llamaagent/issues",
        "Documentation": "https://docs.llamasearch.ai/llamaagent",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=1.8.0",
        "requests>=2.25.0",
        "openai>=0.27.0",
        "tenacity>=8.0.0",
        "tiktoken>=0.3.0",
        "sqlalchemy>=1.4.0",
        "PyYAML>=6.0",
        "aiohttp>=3.8.0",
        "rich>=10.0.0",
        "langchain>=0.0.200",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "mypy>=0.812",
            "isort>=5.9.1",
            "tox>=3.24.0",
        ],
        "tools": [
            "beautifulsoup4>=4.9.0",
            "pillow>=9.0.0",
            "pandas>=1.3.0",
            "numpy>=1.20.0",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
# Updated in commit 5 - 2025-04-04 17:36:19

# Updated in commit 13 - 2025-04-04 17:36:20

# Updated in commit 21 - 2025-04-04 17:36:20

# Updated in commit 29 - 2025-04-04 17:36:21

# Updated in commit 5 - 2025-04-05 14:38:11

# Updated in commit 13 - 2025-04-05 14:38:12

# Updated in commit 21 - 2025-04-05 14:38:12

# Updated in commit 29 - 2025-04-05 14:38:12

# Updated in commit 5 - 2025-04-05 15:24:38

# Updated in commit 13 - 2025-04-05 15:24:38

# Updated in commit 21 - 2025-04-05 15:24:38

# Updated in commit 29 - 2025-04-05 15:24:38

# Updated in commit 5 - 2025-04-05 16:00:21

# Updated in commit 13 - 2025-04-05 16:00:21

# Updated in commit 21 - 2025-04-05 16:00:21

# Updated in commit 29 - 2025-04-05 16:00:21

# Updated in commit 5 - 2025-04-05 17:05:39

# Updated in commit 13 - 2025-04-05 17:05:39

# Updated in commit 21 - 2025-04-05 17:05:39

# Updated in commit 29 - 2025-04-05 17:05:39

# Updated in commit 5 - 2025-04-05 17:37:45

# Updated in commit 13 - 2025-04-05 17:37:45

# Updated in commit 21 - 2025-04-05 17:37:45

# Updated in commit 29 - 2025-04-05 17:37:45

# Updated in commit 5 - 2025-04-05 18:24:30

# Updated in commit 13 - 2025-04-05 18:24:31

# Updated in commit 21 - 2025-04-05 18:24:31

# Updated in commit 29 - 2025-04-05 18:24:31

# Updated in commit 5 - 2025-04-05 18:44:07

# Updated in commit 13 - 2025-04-05 18:44:08

# Updated in commit 21 - 2025-04-05 18:44:08

# Updated in commit 29 - 2025-04-05 18:44:08

# Updated in commit 5 - 2025-04-05 19:10:53

# Updated in commit 13 - 2025-04-05 19:10:54

# Updated in commit 21 - 2025-04-05 19:10:54

# Updated in commit 29 - 2025-04-05 19:10:54

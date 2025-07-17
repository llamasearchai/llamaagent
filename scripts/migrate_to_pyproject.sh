#!/bin/bash
# Migration script for transitioning from requirements*.txt to pyproject.toml
# This script helps users update their installation commands

set -e

echo "🔄 LlamaAgent Dependency Migration Helper"
echo "========================================"
echo ""
echo "This script helps migrate from requirements*.txt files to pyproject.toml"
echo ""

# Check if requirements files exist
if [ -f "requirements.txt" ]; then
    echo "📋 Found requirements.txt"
fi

if [ -f "requirements-dev.txt" ]; then
    echo "📋 Found requirements-dev.txt"
fi

if [ -f "requirements-openai.txt" ]; then
    echo "📋 Found requirements-openai.txt"
fi

echo ""
echo "🚀 Migration Instructions:"
echo ""
echo "Instead of using requirements files, use these commands:"
echo ""
echo "1. Basic installation:"
echo "   OLD: pip install -r requirements.txt"
echo "   NEW: pip install llamaagent"
echo ""
echo "2. Development installation:"
echo "   OLD: pip install -r requirements.txt -r requirements-dev.txt"
echo "   NEW: pip install -e '.[dev]'"
echo ""
echo "3. All features installation:"
echo "   OLD: pip install -r requirements.txt -r requirements-dev.txt -r requirements-openai.txt"
echo "   NEW: pip install -e '.[all,openai-extended]'"
echo ""
echo "4. Specific features:"
echo "   - Vector databases: pip install 'llamaagent[vector]'"
echo "   - Distributed: pip install 'llamaagent[distributed]'"
echo "   - Enterprise: pip install 'llamaagent[enterprise]'"
echo "   - OpenAI extended: pip install 'llamaagent[openai-extended]'"
echo "   - Observability: pip install 'llamaagent[observability]'"
echo ""
echo "📝 Docker users:"
echo "   Dockerfiles have been updated to use pyproject.toml"
echo "   No action needed for docker builds"
echo ""
echo "⚠️  The requirements*.txt files will be deprecated in the next release"
echo ""

# Offer to generate lock files if needed
read -p "Would you like to generate requirements files from pyproject.toml for compatibility? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📦 Generating requirements files from pyproject.toml..."
    
    if command -v uv &> /dev/null; then
        uv pip compile pyproject.toml --output-file requirements.txt
        uv pip compile pyproject.toml --extra dev --output-file requirements-dev.txt
        uv pip compile pyproject.toml --extra all --extra openai-extended --output-file requirements-all.txt
        echo "✅ Generated requirements files using uv"
    elif command -v pip-compile &> /dev/null; then
        pip-compile pyproject.toml -o requirements.txt
        pip-compile pyproject.toml --extra dev -o requirements-dev.txt
        pip-compile pyproject.toml --extra all --extra openai-extended -o requirements-all.txt
        echo "✅ Generated requirements files using pip-compile"
    else
        echo "❌ Neither 'uv' nor 'pip-compile' found. Install with:"
        echo "   pip install uv"
        echo "   # or"
        echo "   pip install pip-tools"
    fi
fi

echo ""
echo "✨ Migration guide complete!"
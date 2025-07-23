#!/usr/bin/env python3
"""Minimal test for the OpenAI API structure without running imports."""

import ast
import re
from pathlib import Path


def extract_endpoints_from_file(file_path):
    """Extract endpoint definitions from the API file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find all FastAPI endpoint decorators
    endpoints = []
    patterns = [
        r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        r'@app\.(get|post|put|delete|patch)\s*\([^)]*path\s*=\s*["\']([^"\']+)["\']',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        for method, path in matches:
            endpoints.append((method.upper(), path))

    return endpoints


def extract_classes_from_file(file_path):
    """Extract Pydantic model classes from the API file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find all class definitions that inherit from BaseModel
    classes = []
    pattern = r'class\s+(\w+)\s*\([^)]*BaseModel[^)]*\)'
    matches = re.findall(pattern, content)
    classes.extend(matches)

    return classes


# Path to the API file
api_file = (
    Path(__file__).parent / "src" / "llamaagent" / "api" / "openai_comprehensive_api.py"
)

if api_file.exists():
    print(" API file found at:", api_file)

    # Extract endpoints
    endpoints = extract_endpoints_from_file(api_file)
    print(f"\n Found {len(endpoints)} endpoints:")

    # Group by category
    endpoint_groups = {
        "Status": [],
        "Chat": [],
        "Reasoning": [],
        "Images": [],
        "Audio": [],
        "Embeddings": [],
        "Moderation": [],
        "Analytics": [],
        "Batch": [],
        "Models": [],
        "Tools": [],
    }

    for method, path in sorted(endpoints):
        category = "Other"
        if path in ["/", "/health", "/budget"]:
            category = "Status"
        elif "/chat" in path:
            category = "Chat"
        elif "/reasoning" in path:
            category = "Reasoning"
        elif "/images" in path:
            category = "Images"
        elif "/audio" in path:
            category = "Audio"
        elif "/embeddings" in path:
            category = "Embeddings"
        elif "/moderation" in path:
            category = "Moderation"
        elif "/usage" in path:
            category = "Analytics"
        elif "/batch" in path:
            category = "Batch"
        elif "/models" in path:
            category = "Models"
        elif "/tools" in path:
            category = "Tools"

        if category not in endpoint_groups:
            endpoint_groups[category] = []
        endpoint_groups[category].append((method, path))

    for category, eps in endpoint_groups.items():
        if eps:
            print(f"\n  {category}:")
            for method, path in eps:
                print(f"    - {method:6} {path}")

    # Extract request/response models
    classes = extract_classes_from_file(api_file)
    print(f"\n Found {len(classes)} Pydantic models:")

    request_models = [c for c in classes if "Request" in c]
    response_models = [c for c in classes if "Response" in c]
    other_models = [c for c in classes if c not in request_models + response_models]

    if request_models:
        print("\n  Request Models:")
        for model in sorted(request_models):
            print(f"    - {model}")

    if response_models:
        print("\n  Response Models:")
        for model in sorted(response_models):
            print(f"    - {model}")

    if other_models:
        print("\n  Other Models:")
        for model in sorted(other_models):
            print(f"    - {model}")

    print("\n API structure analysis complete!")

    # Verify all required endpoints
    required_endpoints = [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/budget"),
        ("GET", "/models"),
        ("POST", "/chat/completions"),
        ("POST", "/reasoning/solve"),
        ("POST", "/images/generate"),
        ("POST", "/embeddings"),
        ("POST", "/moderations"),
        ("POST", "/batch/process"),
        ("GET", "/usage/summary"),
    ]

    found_paths = {path for _, path in endpoints}
    missing = []
    for method, path in required_endpoints:
        if path not in found_paths:
            missing.append(f"{method} {path}")

    if missing:
        print("\n Missing required endpoints:")
        for endpoint in missing:
            print(f"    - {endpoint}")
    else:
        print("\n All required endpoints are implemented!")

else:
    print(" API file not found at:", api_file)

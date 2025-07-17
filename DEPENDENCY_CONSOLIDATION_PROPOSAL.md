# Proposal: Consolidate All Dependencies into pyproject.toml

## Summary

I propose consolidating all project dependencies from the various `requirements*.txt` files into our existing `pyproject.toml` file. This aligns with modern Python packaging standards and will simplify dependency management.

## Current State

We currently maintain dependencies in multiple places:
- `pyproject.toml` - Main project configuration with optional dependency groups
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies  
- `requirements-openai.txt` - OpenAI integration dependencies

## Benefits of Consolidation

1. **Single Source of Truth**: All dependencies in one place reduces confusion and maintenance overhead
2. **Modern Standards**: Follows PEP 517/518/621 packaging standards
3. **Better Dependency Resolution**: pip and other tools can better resolve conflicts
4. **Cleaner Project Structure**: Fewer files in the root directory
5. **Improved User Experience**: Clear installation commands with optional extras

## Proposed Changes

### 1. Add Missing Core Dependencies

Add these essential dependencies currently only in requirements.txt:
```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "alembic>=1.12.0,<2.0.0",  # Database migrations
    "tiktoken>=0.5.0,<1.0.0",   # OpenAI tokenization
    "passlib>=1.7.4,<2.0.0",    # Password hashing
]
```

### 2. Create New Optional Dependency Groups

```toml
[project.optional-dependencies]
# Existing groups remain unchanged...

# New: Extended OpenAI integration support
openai-extended = [
    "pydub>=0.25.1",
    "soundfile>=0.12.1",
    "librosa>=0.10.1",
    "scipy>=1.11.0",
    "opencv-python>=4.8.0",
    "boto3>=1.34.0",
    "minio>=7.2.0",
]

# New: Development profiling tools
dev-profiling = [
    "pytest-benchmark>=4.0.0",
    "line-profiler>=4.1.0", 
    "memory-profiler>=0.61.0",
    "py-spy>=0.3.14",
    "locust>=2.17.0",
]

# New: Kubernetes deployment
kubernetes = [
    "kubernetes>=27.0.0",
]
```

### 3. Update Installation Documentation

Replace current installation instructions:

**Before:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**After:**
```bash
# Basic installation
pip install llamaagent

# Development setup
pip install llamaagent[dev]

# With all features
pip install llamaagent[all]

# Specific features
pip install llamaagent[openai-extended,kubernetes]
```

## Migration Plan

1. **Phase 1**: Add missing dependencies to pyproject.toml
2. **Phase 2**: Update documentation and CI/CD pipelines
3. **Phase 3**: Add deprecation notice to requirements files
4. **Phase 4**: Remove requirements files in next minor release

## Backward Compatibility

To ensure smooth transition:
- Keep requirements files for 1-2 release cycles with deprecation notice
- Add clear migration instructions in CHANGELOG
- Update all documentation and examples

## Questions for Discussion

1. Should we keep requirements.txt as a generated lock file for reproducible installs?
2. Are there any CI/CD workflows that depend on requirements files?
3. Should we add more granular optional dependency groups?

## Action Items

- [ ] Review and approve proposal
- [ ] Update pyproject.toml with missing dependencies
- [ ] Update installation documentation
- [ ] Test all installation methods
- [ ] Update CI/CD pipelines
- [ ] Add deprecation notices
- [ ] Plan removal timeline

---

What are your thoughts on this consolidation? Any concerns or suggestions?

@maintainers @contributors
# PyPI Publishing Instructions for LlamaAgent v0.1.1

## Prerequisites

1. Install required tools:
```bash
pip install --upgrade build twine
```

2. Set up PyPI credentials:
   - Create an account at https://pypi.org
   - Generate an API token from https://pypi.org/manage/account/token/
   - Save token for use with twine

## Build Process (Already Completed)

The distribution files have been built:
- `dist/llamaagent-0.1.1-py3-none-any.whl`
- `dist/llamaagent-0.1.1.tar.gz`

## Publishing to PyPI

### Test PyPI (Recommended First)

1. Upload to Test PyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

2. Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps llamaagent==0.1.1
```

### Production PyPI

1. Upload to PyPI:
```bash
python -m twine upload dist/*
```

2. When prompted:
   - Username: `__token__`
   - Password: Your PyPI API token (starts with `pypi-`)

3. Verify installation:
```bash
pip install llamaagent==0.1.1
```

## Post-Publishing Checklist

- [ ] Verify package page at https://pypi.org/project/llamaagent/
- [ ] Test installation in a clean environment
- [ ] Update GitHub release with PyPI link
- [ ] Update documentation with installation instructions
- [ ] Announce release on social media/forums

## Package Metadata

- **Name**: llamaagent
- **Version**: 0.1.1
- **Author**: Nik Jois
- **Email**: nikjois@llamasearch.ai
- **License**: MIT
- **Python**: >=3.11

## Troubleshooting

If you encounter issues:
1. Ensure you have the latest versions of build and twine
2. Check that all metadata in pyproject.toml is correct
3. Verify the distribution files are not corrupted
4. Use `--verbose` flag with twine for detailed output

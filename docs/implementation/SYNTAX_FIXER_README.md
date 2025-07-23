# Comprehensive Python Syntax Error Fixer

This repository contains two versions of a comprehensive Python syntax error fixing tool designed to automatically identify and fix common syntax errors in Python source code.

## Features

### Version 1 (`comprehensive_syntax_fixer.py`)
- Basic pattern-based syntax error detection and fixing
- Backup creation before modifying files
- Batch processing with configurable batch size
- JSON report generation
- Common fixes include:
  - Missing closing parentheses, brackets, and braces
  - F-string formatting issues
  - Missing colons after control structures
  - Indentation errors
  - Unmatched quotes
  - Trailing commas
  - Assignment operators in if statements

### Version 2 (`comprehensive_syntax_fixer_v2.py`) - Enhanced Version
All features from Version 1, plus:
- **File-specific error handling**: Targets known syntax errors in specific files
- **Multiple iteration fixing**: Attempts to fix errors through up to 5 iterations
- **Enhanced error tracking**: Stores specific error messages and line numbers
- **Improved backup organization**: Creates subdirectories matching source structure
- **Better f-string handling**: More robust detection and fixing of f-string issues
- **Comprehensive reporting**: Includes iteration counts and specific error details
- **Failed files tracking**: Generates a separate list of files that couldn't be fixed

## Usage

### Basic Usage
```bash
# Fix all Python files in the src directory (default)
python3 comprehensive_syntax_fixer_v2.py

# Fix files in a specific directory
python3 comprehensive_syntax_fixer_v2.py /path/to/directory

# Specify batch size
python3 comprehensive_syntax_fixer_v2.py --batch-size 20

# Custom backup directory
python3 comprehensive_syntax_fixer_v2.py --backup-dir my_backups

# Custom report filename
python3 comprehensive_syntax_fixer_v2.py --report my_report.json
```

### Command Line Options
- `directory`: Directory to process (default: src)
- `--batch-size`: Number of files to process in each batch (default: 10)
- `--backup-dir`: Directory for file backups (default: syntax_backups)
- `--report`: Output report file (default: syntax_fix_report.json)

## How It Works

1. **Scanning Phase**: The tool recursively scans the specified directory for Python files
2. **Backup Creation**: Before modifying any file, a timestamped backup is created
3. **Error Detection**: Uses `ast.parse()` to validate Python syntax
4. **Pattern Matching**: Applies regex-based patterns to fix common syntax errors
5. **Iterative Fixing**: For Version 2, attempts multiple iterations if initial fixes don't resolve all errors
6. **Validation**: Re-validates the fixed code to ensure syntax correctness
7. **Restoration**: If fixes fail, the original file is restored from backup
8. **Reporting**: Generates a comprehensive JSON report with all fixes and statistics

## Fix Patterns

The tool handles these common syntax error patterns:

1. **Missing Closing Delimiters**
   - Parentheses: `function_call(arg1, arg2` → `function_call(arg1, arg2)`
   - Brackets: `my_list = [1, 2, 3` → `my_list = [1, 2, 3]`
   - Braces: `my_dict = {"key": "value"` → `my_dict = {"key": "value"}`

2. **F-String Issues**
   - Unclosed f-strings: `f"Hello {name"` → `f"Hello {name}"`
   - Brace mismatches in f-strings

3. **Control Structure Syntax**
   - Missing colons: `if condition` → `if condition:`
   - Empty blocks: Adds `pass` statements where needed

4. **Indentation**
   - Converts tabs to spaces
   - Ensures indentation is in multiples of 4 spaces
   - Fixes unexpected indent errors

5. **Operator Issues**
   - Assignment in if: `if x = 5` → `if x == 5`
   - Trailing commas: `func(arg,)` → `func(arg)`

6. **Quote Matching**
   - Fixes unmatched single and double quotes

## Report Structure

The generated JSON report includes:
- Start and end timestamps
- Total files processed
- Number of files fixed
- Total errors fixed
- List of failed files
- Detailed fixes for each file including:
  - Line numbers
  - Original vs fixed code
  - Type of fix applied
  - Number of iterations needed (Version 2)

## Safety Features

- **Non-destructive**: All original files are backed up before modification
- **Validation**: Fixed code is validated before writing
- **Restoration**: Failed fixes result in automatic restoration from backup
- **Detailed logging**: Console output shows progress and results for each file

## Known Limitations

- Cannot fix all types of syntax errors (e.g., logic errors, missing imports)
- Complex multi-line syntax errors may require manual intervention
- Some edge cases in string literals or comments might be incorrectly modified
- Does not handle encoding issues in source files

## Best Practices

1. **Always review changes**: While the tool is comprehensive, always review the fixes
2. **Use version control**: Run this tool on version-controlled code for easy rollback
3. **Test after fixing**: Run your test suite after applying fixes
4. **Check the report**: Review the detailed report for understanding what was changed
5. **Handle failed files manually**: Files listed in `failed_files.txt` need manual attention

## Examples

### Example 1: Missing Parenthesis
```python
# Before
print("Hello, world"

# After
print("Hello, world")
```

### Example 2: F-string Issue
```python
# Before
message = f"User {user.name} has {len(items} items"

# After
message = f"User {user.name} has {len(items)} items"
```

### Example 3: Missing Colon
```python
# Before
if x > 5
    print("x is greater than 5")

# After
if x > 5:
    print("x is greater than 5")
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## Contributing

To add new fix patterns:
1. Add the regex pattern to the `fix_patterns` list in the `__init__` method
2. Create a corresponding fix method
3. For file-specific fixes (Version 2), add to `_fix_specific_file_errors` method

## License

This tool is provided as-is for educational and development purposes.

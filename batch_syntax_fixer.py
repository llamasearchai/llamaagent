#!/usr/bin/env python3
"""
Batch syntax fixer for common patterns in LlamaAgent framework.
"""

import re
from pathlib import Path
from typing import List

def fix_uuid_parentheses(content: str) -> str:
    """Fix missing closing parentheses in UUID field definitions."""
    patterns = [
        # UUID field definitions missing closing parenthesis
        (r'field\(default_factory=lambda: str\(uuid\.uuid4\(\)\)\)', r'field(default_factory=lambda: str(uuid.uuid4())'),
        (r'str\(uuid\.uuid4\(\)\s*$', r'str(uuid.uuid4())'),
        (r'datetime\.now\(timezone\.utc\)\s*$', r'datetime.now(timezone.utc))'),
        (r'field\(default_factory=lambda: datetime\.now\(timezone\.utc\)\)', r'field(default_factory=lambda: datetime.now(timezone.utc))'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def fix_json_dumps_parentheses(content: str) -> str:
    """Fix missing closing parentheses in json.dumps calls."""
    # Fix json.dumps calls missing closing parenthesis
    content = re.sub(r'json\.dumps\(([^)]+)\s*$', r'json.dumps(\1)', content, flags=re.MULTILINE)
    return content

def fix_function_call_parentheses(content: str) -> str:
    """Fix missing closing parentheses in function calls."""
    # Fix common patterns
    content = re.sub(r'asyncio\.create_task\(([^)]+)\s*$', r'asyncio.create_task(\1)', content, flags=re.MULTILINE)
    content = re.sub(r'list\(([^)]+)\s*$', r'list(\1)', content, flags=re.MULTILINE)
    content = re.sub(r'set\(([^)]+)\s*$', r'set(\1)', content, flags=re.MULTILINE)
    content = re.sub(r'str\(([^)]+)\s*$', r'str(\1)', content, flags=re.MULTILINE)
    
    return content

def fix_bracket_mismatches(content: str) -> str:
    """Fix common bracket mismatches."""
    lines: List[str] = content.split('\n')
    fixed_lines: List[str] = []
    
    for line in lines:
        # Fix hexdigest calls
        if 'hexdigest(' in line and line.count('(') > line.count(')'):
            line = line.replace('hexdigest(', 'hexdigest()')
        
        # Fix other common patterns
        if '.append(' in line and line.count('(') > line.count(')'):
            if not line.rstrip().endswith(')'):
                line = line.rstrip() + ')'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_specific_patterns(content: str) -> str:
    """Fix specific syntax error patterns."""
    # Fix f-string patterns
    content = re.sub(r'f"node_\{str\(uuid\.uuid4\(\)\[:8\]\}"', r'f"node_{str(uuid.uuid4())[:8]}"', content)
    
    # Fix generator expressions that need parentheses
    content = re.sub(r'earliest_finish\.get\(dep\.task_id, timedelta\(\)\) \+ dep\.lag_time\s*for dep in', 
                    r'(earliest_finish.get(dep.task_id, timedelta()) + dep.lag_time for dep in', content)
    
    # Fix mismatched quotes and parentheses
    content = re.sub(r'logger\.error\(f"Error: \{e\}""\) with proper logging', 
                    r'logger.error(f"Error: {e}")', content)
    
    return content

def process_file(file_path: Path) -> bool:
    """Process a single file and fix syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        content = original_content
        
        # Apply fixes
        content = fix_uuid_parentheses(content)
        content = fix_json_dumps_parentheses(content) 
        content = fix_function_call_parentheses(content)
        content = fix_bracket_mismatches(content)
        content = fix_specific_patterns(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main() -> None:
    """Main function to process all Python files."""
    src_dir = Path("src/llamaagent")
    fixed_count = 0
    
    if not src_dir.exists():
        print("src/llamaagent directory not found!")
        return
    
    # Find all Python files
    python_files = list(src_dir.rglob("*.py"))
    
    print(f"Processing {len(python_files)} Python files...")
    
    for file_path in python_files:
        if process_file(file_path):
            fixed_count += 1
    
    print(f"\nCompleted! Fixed {fixed_count} files.")
    
    # Also fix some root level files
    root_files = [
        "comprehensive_fixer.py",
        "production_demo.py", 
        "fastapi_app.py",
        "demo_complete_system.py"
    ]
    
    for root_file in root_files:
        file_path = Path(root_file)
        if file_path.exists():
            if process_file(file_path):
                fixed_count += 1
    
    print(f"Total files fixed: {fixed_count}")

if __name__ == "__main__":
    main() 
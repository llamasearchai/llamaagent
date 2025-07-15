#!/usr/bin/env python3
import os
import ast
import sys

errors = []
for root, dirs, files in os.walk('.'):
    # Skip hidden directories and __pycache__
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f'{filepath}: line {e.lineno}: {e.msg}')
            except Exception as e:
                if 'syntax' in str(e).lower():
                    errors.append(f'{filepath}: {e}')

for error in sorted(errors):
    print(error)
print(f"\nTotal: {len(errors)} files with syntax errors")
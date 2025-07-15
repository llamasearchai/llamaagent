#!/usr/bin/env python3
"""
Quick syntax error fix script for ReactAgent

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import re

def fix_react_agent_syntax():
    """Fix syntax errors in ReactAgent file."""
    
    file_path = "src/llamaagent/agents/react.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix missing parentheses
    fixes = [
        ('return self._wrap_provider(MockProvider()', 'return self._wrap_provider(MockProvider())'),
        ('return asyncio.run(self.execute_task(task_input)', 'return asyncio.run(self.execute_task(task_input))'),
        ('isinstance(self.config.metadata["storage"], dict):', 'isinstance(self.config.metadata["storage"], dict)):'),
    ]
    
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"Fixed: {old} -> {new}")
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("PASS Syntax errors fixed!")

if __name__ == "__main__":
    fix_react_agent_syntax() 
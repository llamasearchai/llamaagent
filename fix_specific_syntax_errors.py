#!/usr/bin/env python3
"""Fix specific syntax errors in remaining files."""

import os

# Dictionary of files and their specific fixes
fixes = {
    "src/llamaagent/cli/config_manager.py": [
        (196, "            logger.error(f\"Failed to save configuration: {e}\"", "            logger.error(f\"Failed to save configuration: {e}\")")
    ],
    "src/llamaagent/cli/role_manager.py": [
        (57, "        )", "})")
    ],
    "src/llamaagent/cli/function_manager.py": [
        (135, "        }", "})")
    ],
    "src/llamaagent/cli/openai_cli.py": [
        (100, "            model_costs.get(model, 0.0", "            model_costs.get(model, 0.0)")
    ],
    "src/llamaagent/cli/diagnostics_cli.py": [
        (77, "    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:", "    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:")
    ],
    "src/llamaagent/cli/code_generator.py": [
        (94, '            "description": request.description,', '            "description": request.description,')
    ],
    "src/llamaagent/reasoning/memory_manager.py": [
        (209, "            if item:", "            if item:")
    ],
    "src/llamaagent/reasoning/chain_engine.py": [
        (53, '    tool_calls: List[ToolCall] = field(default_factory=list)', '    tool_calls: List[ToolCall] = field(default_factory=list)')
    ],
    "src/llamaagent/knowledge/knowledge_generator.py": [
        (651, "        words = text.split(", "        words = text.split()")
    ],
    "src/llamaagent/ml/inference_engine.py": [
        (453, "            )", "})")
    ],
    "src/llamaagent/evolution/adaptive_learning.py": [
        (122, "        py_files = list(self.data_dir.glob(\"*.py\")", "        py_files = list(self.data_dir.glob(\"*.py\"))")
    ],
    "src/llamaagent/prompting/prompt_templates.py": [
        (530, "        return templates", "        return templates")
    ],
    "src/llamaagent/integration/_openai_stub.py": [
        (292, "                )", "                })")
    ],
    "src/llamaagent/optimization/prompt_optimizer.py": [
        (634, "if __name__ == \"__main__\":", "if __name__ == \"__main__\":")
    ]
}

def apply_fixes():
    """Apply specific fixes to files."""
    for file_path, file_fixes in fixes.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"Fixing {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Apply fixes (line numbers are 1-based)
            for line_num, old_content, new_content in file_fixes:
                if line_num <= len(lines):
                    # Adjust for 0-based indexing
                    idx = line_num - 1
                    # Check if we need to fix this line
                    if old_content in lines[idx] or lines[idx].strip() == old_content.strip():
                        # Preserve indentation
                        indent = len(lines[idx]) - len(lines[idx].lstrip())
                        lines[idx] = ' ' * indent + new_content.strip() + '\n'
                        print(f"  Fixed line {line_num}")
                    else:
                        print(f"  Line {line_num} doesn't match expected content")
            
            # Write back
            with open(file_path, 'w') as f:
                f.writelines(lines)
                
            print(f"  Successfully fixed {file_path}")
            
        except Exception as e:
            print(f"  Error fixing {file_path}: {e}")

if __name__ == "__main__":
    apply_fixes()
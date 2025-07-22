#!/usr/bin/env python3
"""
Script to systematically remove all emojis from the LlamaAgent codebase.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import re
import glob
from pathlib import Path

# Common emojis to remove and their replacements
EMOJI_REPLACEMENTS = {
    "SUCCESS:": "SUCCESS:",
    "LAUNCH:": "LAUNCH:",
    "PASS": "PASS",
    "FAIL": "FAIL",
    "FAST:": "FAST:",
    "TOOL:": "TOOL:",
    "TARGET:": "TARGET:",
    "STATS:": "STATS:",
    "IDEA:": "IDEA:",
    "BUILD:": "BUILD:",
    "SEARCH:": "SEARCH:",
    "METRICS:": "METRICS:",
    "LIST:": "LIST:",
    "DESIGN:": "DESIGN:",
    "FEATURE:": "FEATURE:",
    "POWER:": "POWER:",
    "AWARD:": "AWARD:",
    "HOT:": "HOT:",
    "STAR:": "STAR:",
    "ALERT:": "ALERT:",
    "NOTE:": "NOTE:",
    "CODE:": "CODE:",
    "WARNING:": "WARNING:",
    "SKIP:": "SKIP:",
    "TIME:": "TIME:",
    "STOP:": "STOP:",
    "TAG:": "TAG:",
    "GOODBYE:": "GOODBYE:",
    "LOVE:": "LOVE:",
}

def remove_emojis_from_file(file_path: str) -> int:
    """Remove emojis from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        replacements_made = 0
        
        # Replace known emojis
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            if emoji in content:
                content = content.replace(emoji, replacement)
                replacements_made += 1
        
        # Remove any remaining emojis using regex
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        emoji_matches = emoji_pattern.findall(content)
        if emoji_matches:
            content = emoji_pattern.sub('', content)
            replacements_made += len(emoji_matches)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Cleaned {replacements_made} emojis from {file_path}")
            return replacements_made
        
        return 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Main function to remove emojis from all files."""
    print("Starting emoji removal from LlamaAgent codebase...")
    
    # File patterns to process
    patterns = [
        "**/*.py",
        "**/*.md", 
        "**/*.txt",
        "**/*.yml",
        "**/*.yaml",
        "**/*.json",
        "**/*.sh"
    ]
    
    # Directories to skip
    skip_dirs = {
        'venv', '__pycache__', '.git', 'node_modules', 
        'dist', 'build', '.pytest_cache', 'htmlcov'
    }
    
    total_replacements = 0
    files_processed = 0
    
    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            # Skip directories we don't want to process
            if any(skip_dir in file_path for skip_dir in skip_dirs):
                continue
                
            # Skip binary files
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')):
                continue
                
            replacements = remove_emojis_from_file(file_path)
            if replacements > 0:
                total_replacements += replacements
                files_processed += 1
    
    print(f"\nEmoji removal completed!")
    print(f"Files processed: {files_processed}")
    print(f"Total emojis removed: {total_replacements}")
    
    if total_replacements > 0:
        print("\nThe codebase is now emoji-free and professional!")
    else:
        print("\nNo emojis found - codebase is already clean!")

if __name__ == "__main__":
    main() 
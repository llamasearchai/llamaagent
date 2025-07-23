"""CLI module for LlamaAgent."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='llamaagent',
        description='LlamaAgent - Advanced AI Agent Framework',
        epilog='For more information, visit https://github.com/llamasearchai/llamaagent',
    )

    parser.add_argument(
        '--version', action='store_true', help='Show version information'
    )

    parser.add_argument(
        '--master', action='store_true', help='Launch master CLI with advanced features'
    )

    parser.add_argument(
        'command',
        nargs='?',
        choices=['repl', 'api', 'benchmark'],
        help='Command to run',
    )

    args = parser.parse_args()

    if args.version:
        print('LlamaAgent v0.2.4')
        sys.exit(0)

    if args.command == 'repl':
        print('Starting LlamaAgent REPL...')
        # Import and run REPL
        try:
            from .chat_repl import run_repl

            run_repl()
        except ImportError:
            print('REPL module not available')
    elif args.command == 'api':
        print('Starting LlamaAgent API server...')
        # Import and run API
        try:
            from ..api import start_server

            start_server()
        except ImportError:
            print('API module not available')
    elif args.command == 'benchmark':
        print('Running benchmarks...')
        # Import and run benchmarks
        try:
            from ..benchmarks import run_benchmarks

            run_benchmarks()
        except ImportError:
            print('Benchmark module not available')
    else:
        parser.print_help()


__all__ = ['main']

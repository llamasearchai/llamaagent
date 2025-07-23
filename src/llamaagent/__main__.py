"""`python -m llamaagent` - Main entry point with master CLI support."""

import os
import sys
from typing import NoReturn

# Check for master CLI mode
if "--master" in sys.argv or os.environ.get("LLAMAAGENT_MASTER_CLI") == "1":
    try:
        from .cli.master_cli import main as master_main

        def run_master() -> NoReturn:
            """Run master CLI and exit."""
            master_main()
            sys.exit(0)

        run_master()
    except ImportError:
        # Fallback to regular CLI if master CLI is not available
        from .cli import main

        def run_regular() -> NoReturn:
            """Run regular CLI and exit."""
            main()
            sys.exit(0)

        run_regular()
else:
    from .cli import main

    if __name__ == "__main__":
        main()

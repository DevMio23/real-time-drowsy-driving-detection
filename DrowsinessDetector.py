"""
Deprecated entry point.

Use: python main.py

This shim forwards to the unified application for backward compatibility.
"""

import sys

if __name__ == "__main__":
    print(
        "Note: DrowsinessDetector.py is deprecated. "
        "Launching the unified app via main.py...\n"
    )
    from main import main

    sys.exit(main())

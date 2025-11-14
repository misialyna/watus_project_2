#!/usr/bin/env python3
"""
Standalone Watus Interface Starter
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run interface
from standalone_watus_interface import StandaloneWatusInterface

if __name__ == "__main__":
    interface = StandaloneWatusInterface()
    interface.run_interface(
        host=os.getenv('STANDALONE_INTERFACE_HOST', '127.0.0.1'),
        port=int(os.getenv('STANDALONE_INTERFACE_PORT', '5001'))
    )

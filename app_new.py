#!/usr/bin/env python3
"""Main entry point for the Real Estate Agent bot on HuggingFace Spaces."""

import os
import sys

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import and run the main bot function
from telegram_bot import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

#!/usr/bin/env python3
"""
MMU Painter Launcher

Usage:
    python run_mmu_painter.py
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmu_painter_pkg.main import main

if __name__ == '__main__':
    main()

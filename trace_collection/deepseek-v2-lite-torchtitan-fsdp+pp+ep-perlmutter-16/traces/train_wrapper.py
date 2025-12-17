#!/usr/bin/env python3
"""Wrapper script to register custom models before running TorchTitan training."""
import runpy

# Import the registration module to register DeepSeek-V2-Lite model
# This auto-registers the model when imported (see register_deepseek_v2_lite.py)
import register_deepseek_v2_lite

# Run torchtitan.train as __main__
runpy.run_module('torchtitan.train', run_name='__main__')

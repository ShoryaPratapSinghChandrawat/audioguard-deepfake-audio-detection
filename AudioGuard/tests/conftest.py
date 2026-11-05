"""Skip optional research tests when the lightweight baseline environment is used.

CNN/gradio environments install requirements-research.txt; baseline users should
not need the accelerator runtime merely to test the original NumPy pipeline.
"""
import importlib.util
collect_ignore = []
if importlib.util.find_spec("torch") is None:
    collect_ignore.append("test_research.py")

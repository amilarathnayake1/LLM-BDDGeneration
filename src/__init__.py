"""
BDD Generator Research Project

This package contains the core functionality for BDD scenario generation,
evaluation, and analysis using various LLMs.

Package Structure:
- generators/: BDD scenario generators for different LLMs
- evaluators/: Evaluation systems for assessing scenario quality
- comparators/: Metric-based comparison systems (BLEU, METEOR, BERTScore, etc.)
"""

# Make subpackages easily accessible
from . import generators
from . import evaluators
from . import comparators

__all__ = ['generators', 'evaluators', 'comparators']

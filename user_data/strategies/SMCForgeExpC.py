"""
SMCForgeExpC — Experiment C: strong/weak swing HTF TP distinction.

Uses tp_mode=structural_htf with the newly implemented weak-swing HTF columns.
Inherits all other params from SMCForge baseline.
"""

from SMCForge import SMCForge


class SMCForgeExpC(SMCForge):
    """Experiment C: structural_htf TP with strong/weak swing distinction."""
    timeframe = '1h'

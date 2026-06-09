"""
SMCForge4hExp — 4h TF experiment subclass.

Identical to SMCForge but uses 4h as the base timeframe and 1d for the macro
HTF filter. The only purpose of this subclass is to allow freqtrade to load a
separate SMCForge4hExp.json params file so E0 (1h) params are not overwritten.
"""

from SMCForge import SMCForge


class SMCForge4hExp(SMCForge):
    timeframe = '4h'

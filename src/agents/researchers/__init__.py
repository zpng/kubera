"""
Research Agents Package
Contains bull/bear researchers and research judge for Stage 2 (Investment Debate)
"""

from .bull_researcher import BullResearcher
from .bear_researcher import BearResearcher
from .research_judge import ResearchJudge

__all__ = [
    'BullResearcher',
    'BearResearcher',
    'ResearchJudge'
]

""" Files and functionality related to synthetic accessibility """
""" taken from https://github.com/cstein/GB-GA/tree/feature-glide_docking/ """

from typing import List

import numpy as np

from modifiers import gaussian_modifier_clipped

from .neutralize import neutralize_molecules
from .sascorer import calculateScore


def sa_target_score_clipped(m, target: float = 2.230044, sigma: float = 0.6526308) -> float:
    """Computes a synthesizability multiplier for a (range of) synthetic accessibility score(s)
    The return value is between 1 (perfectly synthesizable) and 0 (not synthesizable).
    Based on the work of https://arxiv.org/pdf/2002.07007. Default values from paper.
    :param m: RDKit molecule
    :param target: the target logp value
    :param sigma: the width of the gaussian distribution
    """
    score: float = calculateScore(m)
    return gaussian_modifier_clipped(score, target, sigma)

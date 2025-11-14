"""McCallum-Foster methods
"""

from .tableaus import erk
from .reversible_rk import MCFReversibleERK, AdjointMCFReversibleERK

class MCFEuler(MCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(MCFEuler, self).__init__(kwargs['options']['lam'], erk.A_euler, erk.b_euler, sde=sde, **kwargs)


class AdjointMCFEuler(AdjointMCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointMCFEuler, self).__init__(kwargs['options']['lam'], erk.A_euler, erk.b_euler, sde=sde, **kwargs)

class MCFMidpoint(MCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(MCFMidpoint, self).__init__(kwargs['options']['lam'], erk.A_midpoint, erk.b_midpoint, sde=sde, **kwargs)


class AdjointMCFMidpoint(AdjointMCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointMCFMidpoint, self).__init__(kwargs['options']['lam'], erk.A_midpoint, erk.b_midpoint, sde=sde, **kwargs)

class MCFRK3(MCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(MCFRK3, self).__init__(kwargs['options']['lam'], erk.A_rk3, erk.b_rk3, sde=sde, **kwargs)


class AdjointMCFRK3(AdjointMCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointMCFRK3, self).__init__(kwargs['options']['lam'], erk.A_rk3, erk.b_rk3, sde=sde, **kwargs)

class MCFRK4(MCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(MCFRK4, self).__init__(kwargs['options']['lam'], erk.A_rk4, erk.b_rk4, sde=sde, **kwargs)


class AdjointMCFRK4(AdjointMCFReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointMCFRK4, self).__init__(kwargs['options']['lam'], erk.A_rk4, erk.b_rk4, sde=sde, **kwargs)

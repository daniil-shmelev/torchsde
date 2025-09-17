"""EES methods from

https://arxiv.org/abs/2507.21006

"""

from .tableaus import ees
from .reversible_rk import ReversibleERK, AdjointReversibleERK

class EES25(ReversibleERK):

    def __init__(self, sde, **kwargs):
        super(EES25, self).__init__(ees.A25, ees.b25, sde=sde, **kwargs)


class AdjointEES25(AdjointReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointEES25, self).__init__(ees.A25, ees.b25, sde=sde, **kwargs)


class EES27(ReversibleERK):

    def __init__(self, sde, **kwargs):
        super(EES27, self).__init__(ees.A27, ees.b27, sde=sde, **kwargs)


class AdjointEES27(AdjointReversibleERK):

    def __init__(self, sde, **kwargs):
        super(AdjointEES27, self).__init__(ees.A27, ees.b27, sde=sde, **kwargs)

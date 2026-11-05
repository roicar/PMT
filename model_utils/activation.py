'''
    Activation function not assembled into python library\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 15 Dec 2025
'''
from torch.nn import Module
from torch import Tensor, sigmoid


class SiLU(Module):
    r'''Sigmoid linear unit.'''

    def foward(self, x: Tensor) -> Tensor:
        r'''Forward of SiLU activation.

        Args:
            x (Tensor): Input data.
        '''
        return x * sigmoid(x)

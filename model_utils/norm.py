'''
    Normalization method not assembled into library\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 15 Sep 2025
'''
from torch.nn import Module, Parameter
from torch import empty, Tensor, rsqrt
from torch.nn.init import normal_


class RMSNorm(Module):
    r'''Root mean square layer normalization.'''
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        r'''Initialization of module.

            Args:
                dim (int): Last dimension of data.
                eps (float): Minimal to ensure nonzero denominator (default: 1e-6).
        '''
        super().__init__()
        self.eps = eps
        self.weight = Parameter(empty(dim))
        normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        r'''Forward process of root mean square layer normalization.
        
        Args:
            x (Tensor): Data requiring normalizing.
        '''
        x = x * rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight  # Scaling
        return x

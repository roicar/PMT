'''
    Embedding and position embeddings in Transformer\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 19 Feb 2026
'''
from typing import Final, Optional
from torch.nn import Module, Embedding, Sequential, Linear, ReLU, Parameter
from torch import Tensor, log2, tensor, float32, cat, exp, arange, zeros, outer, sin, cos, empty, long, where, polar, ones_like, maximum
from math import sqrt, log, log2, floor
from torch import log2 as log2_tensor
from torch import log as log_tensor
from torch.nn.init import normal_, constant_
from os import environ

SEQ_LEN: Final = int(environ.get('SEQ_LEN', '768'))
NUM_DIGITS: Final = 32
NUM_OPERANDS: Final = 32


class WordEmbedding(Module):
    r'''Word embedding.'''

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        r'''Initialization of module.

        Args:
            vocab_size (int): Size of vocabulary of tokens in sequences.
            emb_dim (int): Embedding dimension.
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.emb = Embedding(vocab_size, emb_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        r'''Forward process of word embedding.

        Args:
            token_ids (Tensor): Vocabulary indices of tokens in sequences.

        Shapes
            Input:
                Vocabulary indices: `(B, L)` or `(B, 1)`
                    B: the number of sequences
                    L: the length of a sequence
            Output:
                Embeddings of sequences: `(B, L, D)` or `(B, 1, D)`
                    D: embedding dimension
        '''
        return self.emb(token_ids) * sqrt(self.emb_dim)


def alibi_slopes(num_heads: int) -> Tensor:
    r'''Construct ALiBi slopes.'''
    def gp(H: int) -> Tensor:
        r'''Generate a geometric progression: c_h = 2 ^ {-8 * (h + 1) / H}, h = 0 ... H - 1.'''
        s0 = 2 ** (-2 ** -(log2_tensor(tensor(H, dtype=float32)) - 3))
        q = s0
        return tensor([s0 * (q ** i) for i in range(H)])
    log2_res = log2(tensor(num_heads))
    if log2_res.is_integer():
        return gp(num_heads)
    else:
        closest_power = 2 ** floor(log2_res)
        return cat([gp(closest_power), alibi_slopes(num_heads - closest_power)])


def psi(x: Tensor, c: Tensor) -> Tensor:
    r'''Monotonically increasing transformation in FIRE.'''
    return log_tensor(1 + c * x.abs())


class Ftheta(Module):
    r'''Multilayer perceptron in FIRE.'''

    def __init__(self, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = Sequential(
            Linear(1, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_heads)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.unsqueeze(-1))


class PositionEmbedding(Module):
    r'''Position embedding.'''

    def __init__(self, emb_dim: int, pe_scheme: str, decoder_only: bool, num_heads: Optional[int] = None, theta: int = 10000, num_buckets: int = 32, max_distance: int = 128, hidden_dim: int = 32) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            pe_scheme (str): Position embedding scheme (range: ['nope', 'ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire', 'pc']).
            decoder_only (bool): Whether the architecture is decoder-only.
            theta (str): Base of the exponential decay in generating angular frequencies (default: 10,000).
            num_buckets (int): Number of buckets in T5 relative bias (default: 32).
            max_distance (int): Maximum relative distance in T5 relative bias (default: 128).
            hidden_dim (int): Hidden dimension in MLP in FIRE (default: 32).
        '''
        super().__init__()
        if pe_scheme in ['ape-s', 'rpe-s', 'rope']:
            # \theta_k = \Theta^{-\frac{2k}{d}}
            theta_k = exp(arange(0, emb_dim, 2).float()
                          * (-log(theta) / emb_dim))
        if pe_scheme in ['t5', 'alibi', 'fire']:
            # i - j
            rp = arange(SEQ_LEN).unsqueeze(1) - arange(SEQ_LEN)

        # Sinusoid absolute position embedding
        if pe_scheme == 'ape-s':
            ap = arange(SEQ_LEN)
            ape = zeros(SEQ_LEN, emb_dim)
            # Embedded angles for each position (embedding dimension is grouped by 2)
            angle = outer(ap, theta_k)
            ape[:, 0::2] = sin(angle)  # PE(pos, 2k)
            ape[:, 1::2] = cos(angle)  # PE(pos, 2k + 1)
            self.register_buffer('ape', ape)

        # Learnable absolute position embedding
        elif pe_scheme == 'ape-l':
            if decoder_only:
                self.ape = Parameter(empty(SEQ_LEN, emb_dim))
                normal_(self.ape)
            else:
                self.ape_in = Parameter(empty(SEQ_LEN, emb_dim))
                self.ape_out = Parameter(empty(SEQ_LEN, emb_dim))
                normal_(self.ape_in)
                normal_(self.ape_out)

        # Sinusoid relative position embedding
        # Dai Zihang, Yang Zhilin, Yang Yiming, et al. Transformer-XL: Attentive language models beyond a fixed-length context [C]// Proc of the 57th Annual Meeting of the Association for Computational Linguistics. Stroudsburg, PA: Association for Computational Linguistics, 2019.
        # https://aclanthology.org/anthology-files/anthology-files/pdf/P/P19/P19-1285.pdf
        elif pe_scheme == 'rpe-s':
            rp = arange(-SEQ_LEN + 1, SEQ_LEN)
            rpe = zeros(2 * SEQ_LEN - 1, emb_dim)
            angle = outer(rp, theta_k)
            rpe[:, 0::2] = sin(angle)
            rpe[:, 1::2] = cos(angle)
            self.register_buffer('rpe', rpe)

        # Learnable relative position embedding
        # He Pengcheng, Liu Xiaodong, Gao Jianfeng, et al. DeBERTa: Decoding-enhanced BERT with disentangled attention [C]// Proc of the 9th International Conference on Learning Representations. New York: Curran Associates, Inc., 2021.
        # https://arxiv.org/pdf/2006.03654
        elif pe_scheme == 'rpe-l':
            if decoder_only:
                self.rpe = Parameter(empty(2 * SEQ_LEN - 1, emb_dim))
                normal_(self.rpe)
            else:
                self.rpe_in = Parameter(empty(2 * SEQ_LEN - 1, emb_dim))
                self.rpe_out = Parameter(empty(2 * SEQ_LEN - 1, emb_dim))
                normal_(self.rpe_in)
                normal_(self.rpe_out)

        # Learnable relative bias in T5
        # Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text Transformer [J]. Journal of Machine Learning Research, 21(140): 1-67.
        # https://www.jmlr.org/papers/volume21/20-074/20-074.pdf
        elif pe_scheme == 't5':
            half = num_buckets // 2
            dist = rp.abs()
            # Exact bucket indices: -B / 2 + 1 < i - j < B / 2 - 1 corresponds B/2-1...1 0 B/2+1...B-1
            exact_idx = dist + (rp > 0) * half
            # Log-scaled bucket indices: B/2 ~ B-1 determined by |i - j| when |i - j| > B/2-1
            log_idx = (half + (log_tensor(dist.float() / half) /
                       log(max_distance / half) * (half - 1))).to(long)
            # For |i - j| > max_distance, clamp bucket index to B-1
            log_idx = log_idx.clamp(max=num_buckets - 1)
            # Bucket indices
            bucket = where(dist < half, exact_idx, log_idx)
            if decoder_only:
                # Biases for buckets
                self.biases = Embedding(num_buckets, num_heads)
                # T5 relative bias
                b = self.biases(bucket).permute(2, 0, 1)
                self.register_buffer('b', b)
            else:
                self.in_biases = Embedding(num_buckets, num_heads)
                self.out_biases = Embedding(num_buckets, num_heads)
                b_in = self.in_biases(bucket).permute(2, 0, 1)
                b_out = self.out_biases(bucket).permute(2, 0, 1)
                self.register_buffer('b_in', b_in)
                self.register_buffer('b_out', b_out)

        # Attention with linear biases
        # Press O, Smith N, Lewis M. Train short, test long: Attention with linear biases enables input length extrapolation [C]// Proc of the 10th International Conference on Learning Representations. New York: Curran Associates, Inc., 2022.
        # https://arxiv.org/pdf/2108.12409
        elif pe_scheme == 'alibi':
            slopes = -alibi_slopes(num_heads)
            b = rp.unsqueeze(-1) * slopes
            b = b.permute(2, 0, 1)
            self.register_buffer('b', b)

        # Sinusoid rotary position embedding
        # Su Jianlin, Ahmed M, Lu Yu, et al. RoFormer: Enhanced Transformer with rotary position embedding [J]. Neurocomputing, 2024, 568: 127063.
        # https://arxiv.org/pdf/2104.09864
        elif pe_scheme == 'rope':
            # Rotation angles for queries and keys
            ra = outer(arange(SEQ_LEN), theta_k)
            # Convert angles to complex numbers on a unit circle
            ra = polar(ones_like(ra), ra)
            self.register_buffer('ra', ra)

        # Functional interpolation for relative positions
        # Li Shanda, You Chong, Guruganesh G, et al. Functional interpolation for relative positions improves long context Transformers [C]// Proc of the 12th International Conference on Learning Representations. New York: Curran Associates, Inc., 2024: 39574-39599.
        # https://arxiv.org/pdf/2310.04418
        elif pe_scheme == 'fire':
            if decoder_only:
                # Learnable threshold
                self.L = Parameter(empty(1))
                constant_(self.L, 32.0)
                # Straight-Through Estimator
                L = (self.L.round() - self.L).detach() + self.L
                # b(i,j) = f_\theta(\psi(i - j) / \psi(max{L, i})), \psi(x) = log(1 + c * x)
                self.c = Parameter(empty(1))
                constant_(self.c, 1.0)
                x = psi(rp.float(), self.c) / \
                    (psi(maximum(arange(SEQ_LEN).float().unsqueeze(1), L), self.c) + 1e-6)
                self.mlp = Ftheta(num_heads, hidden_dim)
                b = self.mlp(x).permute(2, 0, 1)
                self.register_buffer('b', b)
            else:
                self.L_in = Parameter(empty(1))
                self.L_out = Parameter(empty(1))
                constant_(self.L_in, 32.0)
                constant_(self.L_out, 32.0)
                p = arange(SEQ_LEN).float().unsqueeze(1)
                L_in = (self.L_in.round() - self.L_in).detach() + self.L_in
                L_out = (self.L_out.round() - self.L_out).detach() + self.L_out
                self.c_in = Parameter(empty(1))
                self.c_out = Parameter(empty(1))
                constant_(self.c_in, 1.0)
                constant_(self.c_out, 1.0)
                x_in = psi(rp.float(), self.c_in) / \
                    (psi(maximum(p, L_in), self.c_in) + 1e-6)
                x_out = psi(rp.float(), self.c_out) / \
                    (psi(maximum(p, L_out), self.c_out) + 1e-6)
                self.mlp_in = Ftheta(num_heads, hidden_dim)
                self.mlp_out = Ftheta(num_heads, hidden_dim)
                b_in = self.mlp_in(x_in).permute(2, 0, 1)
                b_out = self.mlp_out(x_out).permute(2, 0, 1)
                self.register_buffer('b_in', b_in)
                self.register_buffer('b_out', b_out)

        # Multilevel position coupling
        # Cho H, Cha J, Bhojanapalli S, et al. Arithmetic Transformers can length-generalize in both operand length and count [C]// Proc of Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Stroudsburg, PA: Association for Computational Linguistics, 2018, 2: 464-468.
        # https://arxiv.org/pdf/2410.15787
        elif pe_scheme == 'pc':
            self.id1_pe = Embedding(NUM_DIGITS + 2, emb_dim, padding_idx=0)
            self.id2_pe = Embedding(NUM_OPERANDS + 1, emb_dim, padding_idx=0)

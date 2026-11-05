'''
    Attention mechanisms in Transformer\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 7 Jan 2026
'''
from typing import Final, Optional, Tuple, Union
from torch.nn import Module, Linear, Dropout, LayerNorm, Parameter
from torch import zeros, Tensor, matmul, view_as_complex, view_as_real, softmax, empty, arange, gather
from model_utils.norm import RMSNorm
from math import sqrt
from torch.nn.init import normal_


class KVCache(Module):
    r'''K & V caches in decoder layers during inference.'''

    def __init__(self, emb_dim: int, num_heads: int, batch_size: int, max_seq_len: int, device: str) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): The number of heads in multi-head attention.
            batch_size (int): Batch size in inference.
            max_seq_len (int): Maximum length of a sequence.
            device (int): On which device the model is loaded.
        '''
        head_dim = emb_dim // num_heads
        self.k_cache = zeros(batch_size, num_heads,
                             max_seq_len, head_dim).to(device)
        self.v_cache = zeros(batch_size, num_heads,
                             max_seq_len, head_dim).to(device)
        # Boudary of cached positions
        self.cached_pos = 0

    def update(self, k: Optional[Tensor], v: Optional[Tensor], b: int) -> None:
        r'''Update row(s) of K and V.

        Args:
            k (Optional[Tensor]): New K row(s).
            v (Optional[Tensor]): New V row(s).
            b (int): The number of sequences in current batch.

        Shape:
            Input:
                Uncached K row(s)
                    Encoder-decoder:
                        1.1 Self attention: `(B, H, D_H)`
                            `B`: the number of sequences
                            `H`: the number of heads
                            `D_H`: embedding dimension per head
                        1.2 Cross attention:
                            1.2.1 Beginning of a batch: `(B, H, N, D_H)`
                                `N`: the length of an input sequence
                            1.2.2 Later: `None`
                    Decoder-only:
                        2.1 Beginning of a batch: `(B, H, N_p + 1, D_H)`
                            `N_p`: the length of a prompt
                        2.2 Later: `(B, H, D_H)`
                Uncached V row(s)
                    Same as uncached K row(s)
        '''
        if k.dim() == 3:
            # 1.1, 2.2
            self.k_cache[:b, :, self.cached_pos] = k
            self.v_cache[:b, :, self.cached_pos] = v
            self.cached_pos += 1
        elif k.dim() == 4:
            # 1.2.1, 2.1
            n = v.size(-2)
            self.k_cache[:b, :, :n] = k
            self.v_cache[:b, :, :n] = v
            self.cached_pos += n

    def __call__(self, b: int, k: Optional[Tensor] = None, v: Optional[Tensor] = None) -> Tuple[Tensor]:
        r'''Update, hit K & V caches and return.

        Args:
            b (int): The number of sequences in current batch.
            k (Optional[Tensor]): New K row(s) (default: None).
            v (Optional[Tensor]): New V row(s) (default: None).

        Shapes:
            Output:
                Current K embeddings
                    Encoder-decoder:
                        1.1 Batched inference:
                            1.1.1 Self attention: `(B, H, L, D_H)`
                                `L`: the length of a current generated sequence
                            1.1.2 Cross attention: `(B, H, N, D_H)`
                        1.2 Single inference:
                            1.2.1 Self attention: `(H, L, D_H)`
                            1.2.2 Cross attention: `(H, N, D_H)`
                    Decoder-only:
                        2.1 Batched inference: `(B, H, N_p + L_r, D_H)`
                            `L_r`: the length of a current response
                    Batched inference: `(H, N_p + L_r, D_H)`
                Current V embeddings
                    Same as current K embeddings
        '''
        if k != None:
            self.update(k, v, b)
        return self.k_cache[:b, :, :self.cached_pos], self.v_cache[:b, :, :self.cached_pos]

    def refresh(self, refresh: bool) -> None:
        r'''Refresh K & V caches at the beginning of inference in a new batch.

        Args:
            refresh (bool): Whether refreshes cache slots.
        '''
        if refresh:
            self.cached_pos = 0


class RPECache(Module):
    r'''Caches for (Q and) K from relative position embeddings ((Qr and) Kr) in self attention during inference.'''

    def __init__(self, emb_dim: int, num_heads: int, max_seq_len: int, device: str, is_encoder: bool, requires_qr: bool = False) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): The number of heads in multi-head attention.
            max_seq_len (int): Maximum length of a sequence.
            device (str): On which device the model is loaded.
            is_encoder (bool): Whether conducts self attention in encoder layers.
            requires_qr (bool): Whether requires a cache space for Qr (default: False).
        '''
        super().__init__()
        self.is_encoder = is_encoder
        self.requires_qr = requires_qr
        head_dim = emb_dim // num_heads
        self.offset = max_seq_len - 1
        if is_encoder:
            if requires_qr:
                self.qr_cache = zeros(
                    num_heads, 2 * max_seq_len - 1, head_dim).to(device)
            self.kr_cache = zeros(
                num_heads, 2 * max_seq_len - 1, head_dim).to(device)
        else:
            # Reverse indexed by 'i - j' (RPE-S) or indexed by 'j - i' (RPE-L)
            self.kr_cache = zeros(num_heads, max_seq_len, head_dim).to(device)
            if requires_qr:
                # Reverse indexed by 'i - j' (RPE-L)
                self.qr_cache = zeros(
                    num_heads, max_seq_len, head_dim).to(device)
            self.bound = self.offset
            self.hit = self.offset

    def update(self, qr: Optional[Union[Tensor, Tuple[Tensor]]], kr: Optional[Union[Tensor, Tuple[Tensor]]]) -> None:
        r'''Update row(s) of (Qr and) Kr.

        Input:
            Uncached Qr values (only in RPE-L):
                Encoder-decoder:
                    1.1 In encoder:
                        1.1.1 Batch/Sequence 0: `(H, 2 * N0 - 1, D_H)`
                            -> add '-(N0 - 1)'...'N0 - 1'
                            `H`: the number of heads
                            `D_H`: embedding dimension per head
                            `N0`: the length of an input sequence (in batch) 0
                        1.1.2 Batch/Sequence b (b > 0):
                            1.1.2.1 If Nb > NB: (`(H, Nb - NB, D_H)`, `(H, Nb - NB, D_H)`)
                                -> add '-(Nb - 1)'...'-NB', 'NB'...'Nb - 1'
                                `Nb`: the length of an input sequence (in batch) b
                                `NB`: the maximum length of an input sequence (among former batches)
                            1.1.2.2 Else: `None`
                    1.2 In decoder:
                        1.2.1 If mb + 1 > MC: `(H, D_H)`
                            -> add 'mb'
                            `mb`: step mb (0 <= mb < Mb)
                            `Mb`: the length of a full output sequence (in batch) b
                            `MC`: the number of cached positions
                        1.2.2 Else: `None`
                Decoder-only:
                    2.1 Sequence 0:
                        2.1.1 Beginning: `(H, N0_p + 1, D_H)`
                            -> add 'N0_p'...'0'
                            `N0_p`: the length of a prompt in sequence 0
                        2.1.2 Later: `(H, D_H)`
                            -> add 'n0'
                            `n0`: step n0 (0 <= n0 < N0_r)
                            `N0_r`: the length of a full response in sequence 0
                    2.2 Sequence b (b > 0):
                        2.2.1 Beginning:
                            2.2.1.1 If Nb_p + 1 > NC: `(H, Nb_p + 1 - NC, D_H)`
                                -> add 'Nb_p'...'NC'
                                `Nb_p`: the length of a prompt in sequence b
                                `NC`: the number of cached positions
                            2.2.1.2 Else: `None`
                        2.2.2 Later:
                            2.2.2.1 If Nb_p + 1 + nb > NC: `(H, D_H)`
                                -> add 'Nb_p + nb'
                                `nb`: step nb (0 < nb < Nb_r)
                                `Nb_r`: the length of a full response in sequence b
                            2.2.2.2 Else: `None`
            Uncached Kr values:
                RPE-S:
                    Same as uncached Qr values in RPE-L
                RPE-L:
                    2.1 Encoder-decoder:
                        2.1.1: In encoder: same as uncached Qr values
                        2.2.2: In decoder: transform positions into opposite numbers in uncached Qr values
                    2.2 Decoder-only: transform positions into opposite numbers in uncached Qr values
        '''
        if isinstance(kr, Tensor):
            if self.is_encoder:
                # 1.1.1
                if self.requires_qr:
                    self.qr_cache[:, self.lhit:self.rhit + 1] = qr
                self.kr_cache[:, self.lhit:self.rhit + 1] = kr
                self.lbound = self.lhit
                self.rbound = self.rhit
            elif kr.dim() == 2:
                # 1.2.2, 2.1.2
                if self.requires_qr:
                    self.qr_cache[:, self.hit] = qr
                self.kr_cache[:, self.hit] = kr
                self.bound -= 1
            else:
                # 2.1.1, 2.2.1.1
                new_bound = self.bound - kr.size(1)
                if self.requires_qr:
                    self.qr_cache[:, new_bound + 1:self.bound + 1] = qr
                self.kr_cache[:, new_bound + 1:self.bound + 1] = kr
                self.bound = new_bound
                self.hit = new_bound
                if self.hit < self.offset:
                    self.hit += 1
        elif isinstance(kr, Tuple):
            # 1.1.2.1
            if self.requires_qr:
                self.qr_cache[:, self.lhit:self.lbound] = qr[0]
                self.qr_cache[:, self.rbound + 1: self.rhit + 1] = qr[1]
            self.kr_cache[:, self.lhit: self.lbound] = kr[0]
            self.kr_cache[:, self.rbound + 1: self.rhit + 1] = kr[1]
            self.lbound = self.lhit
            self.rbound = self.rhit

    def __call__(self, hit_range: Optional[int] = None, qr: Optional[Union[Tensor, Tuple[Tensor]]] = None, kr: Optional[Union[Tensor, Tuple[Tensor]]] = None) -> Union[Tensor, Tuple[Tensor]]:
        r'''Update, hit (Qr and) Kr cache(s) and return.

        Args:
            hit_range (Optional[int]): Length of an input sequence in an encoder layer (default: None).
            kr (Optional[Union[List[Tensor], Tensor]]): Uncached Kr values (default: None).

        Shape:
            Output:
                Current Qr matrices (only in RPE-L):
                    Encoder-decoder:
                        1.1 In encoder: `(H, 2 * Nb - 1, D_H)`
                            -> Positions: '-Nb + 1'...'Nb - 1'
                        1.2 In decoder: `(H, Lb, D_H)`
                            -> Positions: 'Lb - 1'...'0'
                            `Lb`: the length of a current generated sequence at a step in batch/sequence b
                    Decoder-only: `(H, Nb_p + Lb_r, D_H)`
                        -> Positions: 'Nb_p + Lb_r - 1'...'0'
                            `Lb_r`: the length of a current response at a step in batch/sequence b
                Current Kr matrices:
                    In RPE-S:
                        1.1 Decoder in encoder-decoder: `(H, Lb, D_H)`
                            -> Positions: 'Lb - 1'...'0'
                            `Lb`: the length of a current generated sequence at a step in batch/sequence b
                        1.2 Else: same as uncached Qr values in RPE-L
                    In RPE-L:
                        2.1 Encoder-decoder:
                            2.1.1: In encoder: same as current Qr matrices
                            2.2.2: In decoder: transfor positions into opposite numbers in current Qr matrices
                        2.2 Decoder-only: transform positions into opposite numbers in current Qr matrices
        '''
        if self.is_encoder:
            self.lhit = -hit_range + 1 + self.offset
            self.rhit = hit_range - 1 + self.offset
            self.update(qr, kr)
            if self.requires_qr:
                return self.qr_cache[:, self.lhit:self.rhit + 1], self.kr_cache[:, self.lhit:self.rhit + 1]
            return self.kr_cache[:, self.lhit:self.rhit + 1]
        else:
            self.update(qr, kr)
            self.hit -= 1
            if self.requires_qr:
                return self.qr_cache[:, self.hit + 1:], self.kr_cache[:, self.hit + 1:]
            return self.kr_cache[:, self.hit + 1:]

    def reset(self, reset: bool) -> None:
        r'''Reset boundary of hit positions of Kr cache at the beginning of a batch.

            Args:
                reset (bool): Whether resets hit position. 
        '''
        if reset:
            self.hit = self.offset


class UniversalAttention(Module):
    r'''Universal multi-head attention in Transformer supporting addition of relative bias and rotations on query & key embeddings.'''

    def __init__(self, emb_dim: int, num_heads: int, norm_scheme: str, drop_prob: float, is_cross_attn: bool = False) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): The number of heads in multi-head attention.
            norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
            drop_prob (float): Dropout probability.
            is_cross_attn (bool): Whether conducts cross attention (default: False).
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.is_cross_attn = is_cross_attn
        # Pre normalization
        if is_cross_attn:
            self.norm_q = LayerNorm(
                emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
            self.norm_m = LayerNorm(
                emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
        else:
            self.norm = LayerNorm(
                emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
        # Linear layers generating query, key and value embeddings for all heads
        self.q_proj = Linear(emb_dim, emb_dim)
        self.k_proj = Linear(emb_dim, emb_dim)
        self.v_proj = Linear(emb_dim, emb_dim)
        # Linear layer to concatenate results from different heads
        self.out_proj = Linear(emb_dim, emb_dim)
        # Dropouts after computation of attention scores and final output
        self.attn_dropout = Dropout(drop_prob)
        self.out_dropout = Dropout(drop_prob)

    def forward(self, query: Tensor, memory: Optional[Tensor], key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], relative_bias: Optional[Tensor] = None, rotation_angles: Optional[Tensor] = None, requires_cache: bool = False, refresh_cache: bool = False, max_seq_len: Optional[int] = None) -> Tensor:
        r'''Forward process of universal multi-head attention.

        Args:
            query (Tensor): Embeddings of tokens in query sequences.
            memory (Optional[Tensor]): Embeddings of tokens in memory sequences.
            key_padding_mask (Optional[Tensor]): Key padding mask for memory sequences.
            attn_mask (Optional[Tensor]): Attention mask.
            relative_bias (Optional[Tensor]): Relative bias (default: None).
            rotation_angles (Optional[Tensor]): Rotation angles for Q & K (default: None).
            requires_cache (bool): Whether requires K & V caches (default: False).
            refresh_cache (bool): Whether refreshes K & V cache slots (default: False).
            max_seq_len (Optional[int]): The maximum length of a memory sequence (default: None).

        Shapes:
            Input:
                Embeddings of tokens in query sequences:
                    Training: `(B, M, D)`
                        `B`: the number of sequences
                        `M`: the length of a query sequence
                        `D`: embedding dimension
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder: `(B, M, D)`
                            2.1.2 In decoder: `(B, D)`
                        2.2 Decoder-only:
                            2.2.1 Beginning: `(B, N_p + 1, D)`
                                `N_p`: the length of a prompt
                            2.2.2 Later: `(B, D)`
                Embeddings of tokens in memory sequences:
                    Training: `(B, N, D)`
                        `N`: the length of a query sequence (`M` = `N` in self attention)
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 Self attention in encoder:
                                `(B, N, D)`
                            2.1.2 Self attention in decoder:
                                `(B, D)`
                            2.1.3 Cross attention:
                                2.1.3.1 Beginning: `(B, N, D)`
                                2.1.3.2 Later steps: `None`
                        2.2 Decoder-only: Same as query sequences
                Key padding mask for memory sequences:
                    Training:
                        `(B, N)`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 Self attention in encoder: `(B, N)`
                            2.1.2 Self attention in decoder: `None`
                            2.1.3 Cross attention: `(B, N)`
                        2.2 Decoder-only: `(B, N_p + L_r)`
                            `L_r`: the current length of a response
                Attention mask:
                    Training:
                        1.1 Encoder-decoder:
                            1.1.1 Self attention in decoder: `(M, M)`
                            1.1.2 Others: `None`
                        1.2 Decoder-only: `(M, M)`
                    Inference:
                        2.1 Encoder-decoder:
                            `None`
                        2.2 Decoder-only:
                            2.2.1 Beginning: `(N_p + 1, N_p + 1)`
                            2.2.2 Later: `None`
                Relative bias in self attention with T5, ALiBi or FIRE:
                    Training:
                        `(H, M, N)`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder: `(H, M, N)`
                            2.1.2 In decoder: `(H, L)`
                                `L`: the current length of a generated sequence
                        2.2 Decoder-only:
                            2.2.1 Beginning: `(H, N_p + 1)`
                            2.2.2 Later: `(B, H, N_p + L_r)`
                Rotation angles in self attention with RoPE:
                    Training:
                        `(N, D / 2)`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder: `(N, D / 2)`
                            2.1.2 In decoder: `(D / 2)`
                        2.2 Decoder-only:
                            2.2.1 Beginning: `(N_p + 1, D / 2)`
                            2.2.2 Later: `(B, D / 2)`
            Output:
                Result embeddings of tokens in query sequences:
                    Training:
                        `(B, M, D)`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder: `(B, M, D)`
                            2.1.2 In decoder: `(B, D)`
                        2.2 Decoder-only:
                            2.2.1 Beginning: `(B, N_p + 1, D)`
                            2.2.2 Later: `(B, D)`
        '''
        # if requires_cache:
        #     b = query.size(0)
        #     if not hasattr(self, 'kv_cache'):
        #         # Initialize K & V caches
        #         self.kv_cache = KVCache(
        #             self.emb_dim, self.num_heads, b, max_seq_len, next(self.parameters()).device)
        #     else:
        #         self.kv_cache.refresh(refresh_cache)

        #当前 batch 比旧缓存更大时，直接重建缓存当前需要的 max_seq_len 比旧缓存更大时，也重建缓存其他情况继续复用，只清空位置指针
        if requires_cache:
            b = query.size(0)
            cache_device = next(self.parameters()).device
            if not hasattr(self, 'kv_cache'):
                self.kv_cache = KVCache(
                    self.emb_dim, self.num_heads, b, max_seq_len, cache_device)
            else:
                cache_batch = self.kv_cache.k_cache.size(0)
                cache_seq_len = self.kv_cache.k_cache.size(2)
                if b > cache_batch or (max_seq_len is not None and max_seq_len > cache_seq_len):
                    self.kv_cache = KVCache(
                        self.emb_dim, self.num_heads, b, max_seq_len, cache_device)
                else:
                    self.kv_cache.refresh(refresh_cache)


        # Pre normalization
        if self.is_cross_attn:
            query = self.norm_q(query)
            if memory != None:
                memory = self.norm_m(memory)
        else:
            query = self.norm(query)
            if memory != None:
                memory = self.norm(memory)

        # Query embeddings
        Q = self.q_proj(query)

        # Rotate query embeddings
        # Su Jianlin, Ahmed M, Lu Yu, et al. RoFormer: Enhanced Transformer with rotary position embedding [J]. Neurocomputing, 2024, 568: 127063.
        # https://arxiv.org/pdf/2104.09864
        if rotation_angles != None:
            Q = view_as_complex(Q.reshape(*Q.shape[:-1], -1, 2))
            Q = view_as_real(Q * rotation_angles).flatten(-2)

        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.head_dim)
        # Shape: `(B, H, M, D_H)` or (B, H, 1, D_H)`
        Q = Q.unsqueeze(-2) if Q.dim() == 3 else Q.transpose(-2, -3)

        # Key and value embeddings
        if memory != None:
            K, V = self.k_proj(memory), self.v_proj(memory)

            # Rotate key embeddings
            if rotation_angles != None:
                K = view_as_complex(K.reshape(*K.shape[:-1], -1, 2))
                K = view_as_real(K * rotation_angles).flatten(-2)

            K = K.reshape(*K.shape[:-1], self.num_heads, self.head_dim)
            V = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim)
            if requires_cache:
                if K.dim() == 4:
                    K = K.transpose(-2, -3)
                    V = V.transpose(-2, -3)
                K, V = self.kv_cache(b, K, V)
            else:
                K = K.transpose(-2, -3)
                V = V.transpose(-2, -3)
        else:
            K, V = self.kv_cache(b)

        # A = Softmax((QK^\top + B + M^{(pad)} + M^{(attn)}) / \sqrt{d_H})
        attn_scores = matmul(Q, K.transpose(-1, -2))

        if relative_bias != None:
            if relative_bias.dim() == 2:
                relative_bias = relative_bias.unsqueeze(-2)
            attn_scores += relative_bias

        if key_padding_mask != None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(
                1).unsqueeze(1) == 1, float('-1e9'))

        if attn_mask != None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(
                0).unsqueeze(0) == 1, float('-1e9'))

        attn_scores = softmax(attn_scores / sqrt(self.head_dim), dim=-1)
        # Dropout to avoid overfitting
        attn_scores = self.attn_dropout(attn_scores)

        # Z = AV
        attn_output = matmul(attn_scores, V).transpose(-2, -3)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)
        if query.dim() == 2:
            attn_output = attn_output.squeeze(-2)

        # Concate information from different heads
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        # Residual connection
        return query + attn_output


class TransformerXLAttention(Module):
    r'''Multi-head attention with sinusoid relative position embedding.

    Dai Zihang, Yang Zhilin, Yang Yiming, et al. Transformer-XL: Attentive language models beyond a fixed-length context [C]// Proc of the 57th Annual Meeting of the Association for Computational Linguistics. Stroudsburg, PA: Association for Computational Linguistics, 2019.
    https://aclanthology.org/anthology-files/anthology-files/pdf/P/P19/P19-1285.pdf
    '''

    def __init__(self, emb_dim: int, num_heads: int, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): Number of heads used in multi-head attention.
            norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
            drop_prob (float): Dropout probability.
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.norm = LayerNorm(
            emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
        # Linear layers generating query, key and value embeddings from token embeddings for all heads
        self.q_proj = Linear(emb_dim, emb_dim)
        self.kc_proj = Linear(emb_dim, emb_dim)
        self.v_proj = Linear(emb_dim, emb_dim)
        # Linear layer generating key embeddings from relative position embeddings for all heads
        self.kr_proj = Linear(emb_dim, emb_dim)
        # Matrices \mu and \nu to replace query embeddings in the last two terms
        self.mu = Parameter(empty(num_heads, self.head_dim))
        self.nu = Parameter(empty(num_heads, self.head_dim))
        normal_(self.mu)
        normal_(self.nu)
        self.out_proj = Linear(emb_dim, emb_dim)
        self.attn_dropout = Dropout(drop_prob)
        self.out_dropout = Dropout(drop_prob)

    def forward(self, query: Tensor, memory: Optional[Tensor], key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], rpe: Optional[Union[Tensor, Tuple[Tensor]]], teacher_forcing: bool, requires_kv_cache: bool = False, refresh_cache: bool = False) -> Tensor:
        r'''Forward process of self attention in Transformer-XL.

        Args:
            query (Tensor): Embeddings of token(s) in query sequence(s).
            memory (Optional[Tensor]): Embeddings of token(s) in memory sequence(s).
            key_padding_mask (Optional[Tensor]): Key padding mask for memory sequences.
            attn_mask (Optional[Tensor]): Attention mask.
            rpe (Optional[Union[Tensor, Tuple[Tensor]]]): Relative position embeddings.
            teacher_forcing (bool): Whether conducts teacher forcing training or evaluation.
            requires_kv_cache (bool): Whether requires K & V caches (default: False).
            refresh_cache (bool): Whether refreshes K & V cache slots and resets hit positions in Kr cache slots (default: False).
            max_seq_len (Optional[int]): 

        Shapes:
            Input:
                Embeddings of token(s) in query sequence(s):
                    Same as `UniversalAttention`
                Embeddings of token(s) memory sequence(s):
                    Same as `UniversalAttention`
                Relative position embeddings:
                    Training:
                        1.1 Encoder-decoder: `(2 * N - 1, D)`
                        1.2 Decoder-only: `(2 * (N_p + N_r), D)`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder:
                                2.1.1.1 Batch/Sequence 0: `(2 * N0 - 1, D)`
                                    `N0`: the length of an input sequence (in batch) 0
                                2.1.1.2 Batch/Sequence b (b > 0):
                                    2.1.1.2.1 If Nb > NB: (`(Nb - NB, D)`, `(Nb - NB, D)`)
                                        `Nb`: the length of an input sequence (in batch) b
                                        `NB`: the maximum length of an input sequence (among former batches)
                                    2.1.1.2.2 Else: `None`requires_cache
                            2.1.2 In decoder:
                                2.1.2.1 If mb + 1 > MC: `(D)`
                                    `mb`: step mb (0 <= mb < Mb)
                                    `Mb`: the length of a full output sequence (at batch) b (b >= 0)
                                    `MC`: the number of cached positions
                                2.1.2.2 Else: `None`
                        2.2 Decoder-only:
                            2.2.1 Sequence 0:
                                2.2.1.1 Beginning: `(N0_p + 1, D)`
                                    `N0_p`: the length of a prompt in sequence 0
                                2.2.1.2 Later: `(D)`
                            2.2.2 Sequence b (b > 0):
                                2.2.2.1 Beginning:
                                    2.2.2.1.1 If Nb_p + 1 > NC: `(Nb_p + 1 - NC, D)`
                                        `Nb_p`: the length of a prompt in sequence b
                                        `NC`: the number of cached positions
                                    2.2.2.1.2 Else: `None`
                                2.2.2.2 Later:
                                    2.2.2.2.1 If Nb_p + 1 + nb > NC: `(D)`
                                        `nb`: step nb (0 < nb < Nb_r)
                                        `Nb_r`: the length of a full response in sequence b
                                    2.2.2.2.2 Else: `None`
                Key padding mask for memory sequences:
                    Same as  `UniversalAttention`
                Attention mask:
                    Same as `UniversalAttention`
            Output:
                Result embeddings of token(s) in query sequence(s):
                    Same as `UniversalAttention`
        '''
        requires_rpe_cache = not teacher_forcing
        b = 0 if query.dim() == 1 else query.size(0)
        if requires_kv_cache:
            if not hasattr(self, 'kv_cache'):
                # Initialize K & V caches
                self.kv_cache = KVCache(
                    self.emb_dim, self.num_heads, b, next(self.parameters()).device)
            else:
                self.kv_cache.refresh(refresh_cache)
        if requires_rpe_cache:
            hit_range = -1 if requires_kv_cache else query.size(1)
            if not hasattr(self, 'rpe_cache'):
                # Initialize Kr cache
                self.rpe_cache = RPECache(self.emb_dim, self.num_heads, next(
                    self.parameters()).device, query.dim() == 3)
            elif requires_kv_cache:
                self.rpe_cache.reset(refresh_cache)

        # Query embeddings from token embeddings (content)
        query = self.norm(query)
        Q = self.q_proj(query)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.head_dim)
        Q = Q.unsqueeze(-2) if requires_kv_cache else Q.transpose(-2, -3)
        # Key and value embeddings from content
        if memory != None:
            memory = self.norm(memory)
            Kc, V = self.kc_proj(memory), self.v_proj(memory)
            Kc = Kc.reshape(*Kc.shape[:-1], self.num_heads, self.head_dim)
            V = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim)
            if requires_kv_cache:
                if memory.dim() != query.dim():
                    Kc = Kc.transpose(-2, -3)
                    V = V.transpose(-2, -3)
                Kc, V = self.kv_cache(b, Kc, V)
            else:
                Kc = Kc.transpose(-2, -3)
                V = V.transpose(-2, -3)
        else:
            Kc, V = self.kv_cache(b)
        # Content-to-content attention (1st term)
        attn_cc = matmul(Q, Kc.transpose(-1, -2))

        # Key embeddings from relative position embeddings
        if isinstance(rpe, Tensor):
            Kr = self.kr_proj(rpe)
            Kr = Kr.reshape(*Kr.shape[:-1], self.num_heads, self.head_dim)
            if rpe.dim() == 2:
                Kr = Kr.transpose(0, 1)
            if requires_rpe_cache:
                Kr = self.rpe_cache(hit_range, kr=Kr)
        elif isinstance(rpe, Tuple):
            lKr = self.kr_proj(rpe[0])
            lKr = lKr.reshape(*lKr.shape[:-1], self.num_heads, self.head_dim)
            lKr = lKr.transpose(0, 1)
            rKr = self.kr_proj(rpe[1])
            rKr = rKr.reshape(*rKr.shape[:-1], self.num_heads, self.head_dim)
            rKr = rKr.transpose(0, 1)
            Kr = self.rpe_cache(hit_range, kr=(lKr, rKr))
        else:
            Kr = self.rpe_cache(hit_range)

        # Content-to-position attention (2nd term)
        attn_cp = matmul(Q, Kr.transpose(-1, -2))
        if not requires_kv_cache:
            n = Q.size(-2)
            # i - j
            shifts = ((arange(n - 1, -n, -1) + arange(n).unsqueeze(1)) %
                      (2 * n - 1)).to(next(self.parameters()).device)
            if b:
                attn_cp = gather(
                    attn_cp, dim=-1, index=shifts.expand(b, self.num_heads, -1, -1))[:, :, :, :n]
            else:
                attn_cp = gather(
                    attn_cp, dim=-1, index=shifts.expand(self.num_heads, -1, -1))[:, :, :n]

        # Weighted content term (3rd term)
        attn_wc = matmul(self.mu.unsqueeze(1), Kc.transpose(-1, -2))

        # Weighted position term (4th term)
        attn_wp = matmul(self.nu.unsqueeze(1), Kr.transpose(-1, -2))
        if not requires_kv_cache:
            attn_wp = attn_wp.expand(self.num_heads, n, -1)
            attn_wp = gather(
                attn_wp, dim=-1, index=shifts.expand(self.num_heads, -1, -1))[:, :, :n]

        # Sum the above 4 terms
        attn_scores = attn_cc + attn_cp + attn_wc + attn_wp

        if key_padding_mask != None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(
                1).unsqueeze(1) == 1, float('-1e9'))
        if attn_mask != None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(
                0).unsqueeze(0) == 1, float('-1e9'))
        attn_scores = softmax(attn_scores / sqrt(4 * self.head_dim), dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        attn_output = matmul(attn_scores, V).transpose(-2, -3)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)
        if requires_kv_cache:
            attn_output = attn_output.squeeze(-2)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        return query + attn_output


class DisentangledAttention(Module):
    r'''Multi-head attention with learnable relative position embedding.

    He Pengcheng, Liu Xiaodong, Gao Jianfeng, et al. DeBERTa: Decoding-enhanced BERT with disentangled attention [C]// Proc of the 9th International Conference on Learning Representations. New York: Curran Associates, Inc., 2021.
    https://arxiv.org/pdf/2006.03654
    '''

    def __init__(self, emb_dim: int, num_heads: int, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of module.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int): Number of heads used in multi-head attention.
            norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
            drop_prob (float): Dropout probability.
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.norm = LayerNorm(
            emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
        self.qc_proj = Linear(emb_dim, emb_dim)
        self.kc_proj = Linear(emb_dim, emb_dim)
        self.v_proj = Linear(emb_dim, emb_dim)
        # Linear layer generating query and key embeddings from relative position embeddings for all heads
        self.qr_proj = Linear(emb_dim, emb_dim)
        self.kr_proj = Linear(emb_dim, emb_dim)
        self.out_proj = Linear(emb_dim, emb_dim)
        self.attn_dropout = Dropout(drop_prob)
        self.out_dropout = Dropout(drop_prob)

    def forward(self, query: Tensor, memory: Tensor, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], rpe: Optional[Union[Tensor, Tuple[Tensor]]], teacher_forcing: bool, requires_kv_cache: bool = False, refresh_cache: bool = False) -> Tensor:
        r'''Forward process of disentangled multi-head attention.

        Args:
            query (Tensor): Embeddings of token(s) in query sequence(s).
            memory (Tensor): Embeddings of token(s) in memory sequence(s).
            key_padding_mask (Tensor): Key padding mask for memory sequences.
            attn_mask (Optional[Tensor]): Attention mask for query sequences.
            rpe (Optional[Union[Tensor, Tuple[Tensor]]]): Relative position embeddings.
            teacher_forcing (bool): Whether conducts teacher forcing training or evaluation.
            requires_kv_cache (bool): Whether requires K & V caches (default: False).
            refresh_cache (bool): Whether refreshes cache slots or resets hit position (default: False).

        Shapes:
            Input:
                Embeddings of token(s) in query sequence(s):
                    Same as `TransformerXLAttention`
                Embeddings of token(s) in memory sequence(s):
                    Same as `TransformerXLAttention`
                Relative position embeddings:
                    Training:
                        Same as `TransformerXLAttention`
                    Inference:
                        2.1 Encoder-decoder:
                            2.1.1 In encoder:
                                Same as `TransformerXLAttention`
                            2.1.2 In decoder:
                                2.1.2.1 If mb + 1 > MC: `((D), (D))`
                                    `mb`: step mb (0 <= mb < Mb)
                                    `Mb`: the length of a full output sequence (at batch) b (b >= 0)
                                    `MC`: the number of cached positions
                                2.1.2.2 Else: `None`
                        2.2 Decoder-only:
                            2.2.1 Sequence 0:
                                2.2.1.1 Beginning: `((N0_p + 1, D), (N0_p + 1, D))`
                                    `N0_p`: the length of a prompt in sequence 0
                                2.2.1.2 Later: `((D), (D))`
                            2.2.2 Sequence b (b > 0):
                                2.2.2.1 Beginning:
                                    2.2.2.1.1 If Nb_p + 1 > NC: `((Nb_p + 1 - NC, D), (Nb_p + 1 - NC, D))`
                                        `Nb_p`: the length of a prompt in sequence b
                                        `NC`: the number of cached positions
                                    2.2.2.1.2 Else: `None`
                                2.2.2.2 Later:
                                    2.2.2.2.1 If Nb_p + 1 + nb > NC: `((D), (D))`
                                        `nb`: step nb (0 < nb < Nb_r)
                                        `Nb_r`: the length of a full response in sequence b
                                    2.2.2.2.2 Else: `None`
                Key padding mask for memory sequences:
                    Same as `UniversalAttention`
                Attention mask:
                    Same as `UniversalAttention`
            Output:
                Result embeddings of token(s) in query sequence(s):
                    Same as `UniversalAttention`
        '''
        requires_rpe_cache = not teacher_forcing
        b = 0 if query.dim() == 1 else query.size(0)
        if requires_kv_cache:
            if not hasattr(self, 'kv_cache'):
                # Initialize K & V caches
                self.kv_cache = KVCache(
                    self.emb_dim, self.num_heads, b, next(self.parameters()).device)
            else:
                self.kv_cache.refresh(refresh_cache)
        if requires_rpe_cache:
            hit_range = -1 if requires_kv_cache else query.size(1)
            if not hasattr(self, 'rpe_cache'):
                # Initialize Kr cache
                self.rpe_cache = RPECache(self.emb_dim, self.num_heads, next(
                    self.parameters()).device, query.dim() == 3, True)
            elif requires_kv_cache:
                self.rpe_cache.reset(refresh_cache)

        # Query embeddings from token embeddings (content)
        query = self.norm(query)
        Qc = self.qc_proj(query)
        Qc = Qc.reshape(*Qc.shape[:-1], self.num_heads, self.head_dim)
        Qc = Qc.unsqueeze(-2) if requires_kv_cache else Qc.transpose(-2, -3)
        # Key and value embeddings from content
        if memory != None:
            memory = self.norm(memory)
            Kc, V = self.kc_proj(memory), self.v_proj(memory)
            Kc = Kc.reshape(*Kc.shape[:-1], self.num_heads, self.head_dim)
            V = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim)
            if requires_kv_cache:
                if memory.dim() != query.dim():
                    Kc = Kc.transpose(-2, -3)
                    V = V.transpose(-2, -3)
                Kc, V = self.kv_cache(b, Kc, V)
            else:
                Kc = Kc.transpose(-2, -3)
                V = V.transpose(-2, -3)
        else:
            Kc, V = self.kv_cache(b)
        attn_cc = matmul(Qc, Kc.transpose(-1, -2))

        # Query and key embeddings from relative position embeddings
        if isinstance(rpe, Tensor):
            Qr = self.qr_proj(rpe)
            Qr = Qr.reshape(*Qr.shape[:-1], self.num_heads, self.head_dim)
            Kr = self.kr_proj(rpe)
            Kr = Kr.reshape(*Kr.shape[:-1], self.num_heads, self.head_dim)
            Qr = Qr.transpose(0, 1)
            Kr = Kr.transpose(0, 1)
            if requires_rpe_cache:
                Qr, Kr = self.rpe_cache(hit_range, Qr, Kr)
        elif isinstance(rpe, Tuple):
            if requires_kv_cache:
                Qr = self.qr_proj(rpe[0])
                Qr = Qr.reshape(*Qr.shape[:-1], self.num_heads, self.head_dim)
                Kr = self.kr_proj(rpe[1])
                Kr = Kr.reshape(*Kr.shape[:-1], self.num_heads, self.head_dim)
                if Qr.dim() == 3:
                    Qr = Qr.transpose(0, 1)
                    Kr = Kr.transpose(0, 1)
                Qr, Kr = self.rpe_cache(hit_range, Qr, Kr)
            else:
                lQr = self.qr_proj(rpe[0])
                lQr = lQr.reshape(
                    *lQr.shape[:-1], self.num_heads, self.head_dim)
                lQr = lQr.transpose(0, 1)
                rQr = self.qr_proj(rpe[1])
                rQr = rQr.reshape(
                    *rQr.shape[:-1], self.num_heads, self.head_dim)
                rQr = rQr.transpose(0, 1)
                lKr = self.kr_proj(rpe[0])
                lKr = lKr.reshape(
                    *lKr.shape[:-1], self.num_heads, self.head_dim)
                lKr = lKr.transpose(0, 1)
                rKr = self.kr_proj(rpe[1])
                rKr = rKr.reshape(
                    *rKr.shape[:-1], self.num_heads, self.head_dim)
                rKr = rKr.transpose(0, 1)
                Qr, Kr = self.rpe_cache(hit_range, (lQr, rQr), (lKr, rKr))
        else:
            Qr, Kr = self.rpe_cache(hit_range)

        # Content-to-position attention
        attn_cp = matmul(Qc, Kr.transpose(-1, -2))
        if not requires_kv_cache:
            n = Qc.size(-2)
            # j - i
            kr_shifts = ((arange(2 * n - 1) + arange(n - 1, -1, -1).unsqueeze(1)) %
                         (2 * n - 1)).to(next(self.parameters()).device)
            if b:
                attn_cp = gather(
                    attn_cp, dim=-1, index=kr_shifts.expand(b, self.num_heads, -1, -1))[:, :, :, :n]
            else:
                attn_cp = gather(
                    attn_cp, dim=-1, index=kr_shifts.expand(self.num_heads, -1, -1))[:, :, :n]

        # Position-to-content attention
        attn_pc = matmul(Qr, Kc.transpose(-1, -2))
        if requires_kv_cache:
            if query.dim() == 1:
                attn_pc = attn_pc[:, -1:]
            else:
                attn_pc = attn_pc[:, :, -1:]
        else:
            # i - j
            qr_shifts = ((arange(n - 1, -1, -1) + arange(2 * n - 1).unsqueeze(1)) %
                         (2 * n - 1)).to(next(self.parameters()).device)
            if b:
                attn_pc = gather(
                    attn_pc, dim=-2, index=qr_shifts.expand(b, self.num_heads, -1, -1))[:, :, :n]
            else:
                attn_pc = gather(
                    attn_pc, dim=-2, index=qr_shifts.expand(self.num_heads, -1, -1))[:, :n]

        # Sum the above 3 terms
        attn_scores = attn_cc + attn_cp + attn_pc
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(
                1).unsqueeze(1) == 1, float('-1e9'))
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(
                0).unsqueeze(0) == 1, float('-1e9'))
        attn_scores = softmax(attn_scores / sqrt(3 * self.head_dim), dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        attn_output = matmul(attn_scores, V).transpose(-2, -3)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)
        if requires_kv_cache:
            attn_output = attn_output.squeeze(-2)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        return query + attn_output

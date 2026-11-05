'''
    Transformer models for polynomial arithmatics\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 7 Jan 2026
'''
from typing import Final, Optional, Union, Tuple, List
from torch.nn import Module, LayerNorm, ReLU, GELU, Linear, Dropout, ModuleList
from model_utils.norm import RMSNorm
from model_utils.activation import SiLU
from torch import Tensor, triu, ones
from model_utils.attention import TransformerXLAttention, DisentangledAttention, UniversalAttention
from model_utils.embedding import WordEmbedding, PositionEmbedding

# Maximum number of tokens in a sequence
SEQ_LEN: Final = 768


class FeedForward(Module):
    r'''Feed forward neural network.'''

    def __init__(self, num_layers: int, emb_dim: int, hidden_dim: int, activate_fn: str, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of module.

        Args:
            num_layers (int): The number of layers in feed forward network.
            emb_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            activate_fn (str): Activation function (range: ['relu', 'gelu', 'silu']).
            norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
            drop_prob (float): Dropout probability.
        '''
        super().__init__()

        # Pre normalization
        self.norm = LayerNorm(
            emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)

        Activation = ReLU() if activate_fn == 'relu' else GELU(
        ) if activate_fn == 'gelu' else SiLU()

        # Multi-layer perceptron
        self.layers = ModuleList(
            [Linear(emb_dim, hidden_dim), Activation, Dropout(drop_prob)])
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            self.layers.append(Activation)
            self.layers.append(Dropout(drop_prob))
        self.layers.append(Linear(hidden_dim, emb_dim))

        # Dropout before residual connection
        self.dropout = Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        r'''Forward process of feed forward network.'''
        # Pre normalization
        x_prime = self.norm(x)
        for i in range(len(self.layers)):
            x_prime = self.layers[i](x_prime)

        return x + self.dropout(x_prime)


class TransformerEncoderLayer(Module):
    r'''Encoder layer in Transformer.'''

    def __init__(self, pe_scheme: str, emb_dim: int, num_heads: int, num_ffn_layers: int, hidden_dim: int, activate_fn: str, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of encoder.

            Args:
                pe_scheme (str): Position embedding scheme (range: ['nope', 'ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire']).
                emb_dim (int): Embedding dimension.
                num_heads (int): The number of heads in multi-head attention.
                num_ffn_layers (int): Number of layers in feed forward module.
                hidden_dim (int): Hidden dimension in feed forward module.
                activate_fn (str): Activation function (range: ['relu', 'gelu', 'silu']).
                norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
                drop_prob (float): Dropout probability.
        '''
        super().__init__()
        self.pe_scheme = pe_scheme
        # Multi-head attention
        if pe_scheme == 'rpe-s':
            self.mha = TransformerXLAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)
        elif pe_scheme == 'rpe-l':
            self.mha = DisentangledAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)
        else:
            self.mha = UniversalAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)

        # Feed forward module
        self.ffn = FeedForward(num_ffn_layers, emb_dim,
                               hidden_dim, activate_fn, norm_scheme, drop_prob)

    def forward(self, x: Tensor, mask: Tensor, rpe: Optional[Union[Tensor, Tuple[Tensor]]], b: Optional[Tensor], ra: Optional[Tensor]) -> Tensor:
        r'''Forward process of an encoder layer in Transformer.

        Args:
            x (Tensor): Embeddings of input sequence(s).
            mask (Tensor): Key padding mask for input sequences.
            rpe (Optional[Union[Tensor, Tuple[Tensor]]]): Relative position embeddings.
            b (Optional[Tensor]): Relative bias.
            ra (Optional[Tensor]): Rotation angles for Q & K.

        Shapes:
            Input:

                Embeddings of input sequence(s):
                    Training/Batched inference: `(B, N, D)`
                        `B`: the number of sequences
                        `N`: the length of an input sequence
                        `D`: embedding dimension
                    Single inference: `(N, D)`
                Key padding mask for input sequences:
                    Training/Batched inference: `(B, N)`
                    Single inference: `None`
                Relative position embedding (RPE-S, RPE-L):
                    Training:
                        `(2 * N - 1, D)`
                    Inference:
                        2.1 Batch/Sequence 0: `(2 * N0 - 1, D)`
                                `N0`: the length of an input sequence (in batch) 0
                        2.2 Batch/Sequence b (b > 0):
                            2.2.1 If Nb > NB: (`(Nb - NB, D)`, `(Nb - NB, D)`)
                                `Nb`: the length of an input sequence (in batch) b
                                `NB`: the maximum length of an input sequence (among former batches)
                            2.2.2 Else: `None`
                Relative bias (T5, ALiBi, FIRE): `(H, N, N)`
                    `H`: the number of heads
                Rotation angles of Q & K (RoPE): `(N, D / 2)`
            Output:
                Result embeddings of input sequence(s):
                    Training/Batched inference: `(B, N, D)`
                    Single inference: `(N, D)`
        '''
        # Self attention
        if self.pe_scheme in ['rpe-s', 'rpe-l']:
            x = self.mha(x, x, mask, None, rpe)
        else:
            x = self.mha(x, x, mask, None, b, ra)
        # Feed forward network
        x = self.ffn(x)

        return x


class TransformerDecoderLayer(Module):
    r'''Decoder layer in encoder-decoder Transformer.'''

    def __init__(self, pe_scheme: str, emb_dim: int, num_heads: int, num_ffn_layers: int, hidden_dim: int, activate_fn: str, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of decoder.

            Args:
                pe_scheme (str): Position embedding scheme (range: ['nope', 'ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire']).
                emb_dim (int): Embedding dimension.
                num_heads (int): The number of heads in multi-head attention.
                num_ffn_layers (int): The number of layers in feed forward module.
                hidden_dim (int): Hidden dimension in feed forward module.
                activate_fn (str): Activation function (range: ['relu', 'gelu', 'silu']).
                norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
                drop_prob (float): Dropout probability.
        '''
        super().__init__()
        self.pe_scheme = pe_scheme
        self.mha = ModuleList([])
        # Self attention
        if pe_scheme == 'rpe-s':
            self.mha.append(TransformerXLAttention(
                emb_dim, num_heads, norm_scheme, drop_prob))
        elif pe_scheme == 'rpe-l':
            self.mha.append(DisentangledAttention(
                emb_dim, num_heads, norm_scheme, drop_prob))
        else:
            self.mha.append(UniversalAttention(
                emb_dim, num_heads, norm_scheme, drop_prob))
        # Cross attention
        self.mha.append(UniversalAttention(
            emb_dim, num_heads, norm_scheme, drop_prob, True))

        self.ffn = FeedForward(num_ffn_layers, emb_dim,
                               hidden_dim, activate_fn, norm_scheme, drop_prob)

    def forward(self, x_out: Tensor, out_mask: Optional[Tensor], x_in: Optional[Tensor], in_mask: Tensor, attn_mask: Optional[Tensor], rpe: Optional[Union[Tensor, Tuple[Tensor]]], b: Optional[Tensor], ra: Optional[Tensor], teacher_forcing: bool, refresh_cache: bool, max_seq_len: Optional[List]) -> Tensor:
        r'''Forward process of a decoder layer in encoder-decoder Transformer.

        Args:
            x_out (Tensor): Embeddings of output sequences or focused tokens.
            out_mask (Optional[Tensor]): Key padding mask for output sequences.
            x_in (Optional[Tensor]): Embeddings of input sequences.
            in_mask (Tensor): Key padding mask for input sequences.
            attn_mask (Optional[Tensor]): Attention mask.
            rpe (Optional[Union[Tensor, Tuple[Tensor]]]): Relative position embeddings in output sequences.
            b (Optional[Tensor]): Relative bias in self attention.
            ra (Optional[Tensor]): Rotation angles for Q & K in self attention.
            teacher_forcing (bool): Whether conducts teacher forcing training or evaluation.
            refresh_cache (bool): Whether refreshes K & V cache slots (and resets hit positions in (Qr &) Kr cache slots).
            max_seq_len (Optional[List]): The maximum lengths of an input sequence and an output sequence.

        Shapes:
            Input:
                Embeddings of output sequences or focused tokens
                    Training: `(B, M, D)`
                        `B`: the number of sequences
                        `M`: the length of a full output sequence
                        `D`: embedding dimension
                    Inference: `(B, D)`
                Key padding mask in self attention
                    Training: `(B, M)`
                    Inference: `None`
                Embeddings of input sequence(s)
                    Training: `(B, N, D)`
                        `N`: the length of an input sequence
                    Inference:
                        2.1 Beginning of a batch: `(B, N, D)`
                        2.2 Later steps: `None`
                Key padding mask in cross attention
                    `(B, N)`
                Attention mask
                    Training: `(M, M)`
                    Inference: `None`
                Relative position embeddings in self attention (RPE-S, RPE-L)
                    RPE-S:
                        1.1 Training:
                            `(2 * M - 1, D)`
                        1.2 Inference:
                            1.2.1 If mb + 1 > MC: `(D)`
                                `mb`: step mb (0 <= mb < Mb)
                                `Mb`: the length of a full output sequence (at batch) b (b >= 0)
                                `MC`: the number of cached positions
                            1.2.2 Else: `None`
                    RPE-L:
                        2.1 Training:
                            `(2 * M - 1, D)`
                        2.2 Inference:
                            2.2.1 If mb + 1 > MC: `((D), (D))`
                            2.2.2 Else: `None`
                Relative bias in self attention (T5, ALiBi, FIRE)
                    Training: `(H, M, M)`
                        `H`: the number of heads
                    Inference: `(H, L)`
                        `L`: the length of a current generated sequence
                Rotation angles for Q & K in self attention (RoPE)
                    Training: `(M, D / 2)`
                    Inference: `(D / 2)`
            Output:
                Result embeddings of output sequences or next tokens
                    Training: `(B, M, D)`
                    Inference: `(B, D)`
        '''
        requires_cache = not teacher_forcing
        max_in_seq_len, max_out_seq_len = (
            None, None) if max_seq_len is None else max_seq_len
        # Self attention
        if self.pe_scheme in ['rpe-s', 'rpe-l']:
            x_out = self.mha[0](x_out, x_out, out_mask, attn_mask, rpe,
                                teacher_forcing, requires_cache, refresh_cache, max_out_seq_len)
        else:
            x_out = self.mha[0](x_out, x_out, out_mask, attn_mask,
                                b, ra, requires_cache, refresh_cache, max_out_seq_len)
        # Cross attention
        x_out = self.mha[1](x_out, x_in, in_mask, None, None,
                            None, requires_cache, refresh_cache, max_in_seq_len)
        # Feed forward
        x_out = self.ffn(x_out)

        return x_out


class TransformerDecoderOnlyLayer(Module):
    r'''Decoder layer in decoder-only Transformer.'''

    def __init__(self, pe_scheme: str, emb_dim: int, num_heads: int, num_ffn_layers: int, hidden_dim: int, activate_fn: str, norm_scheme: str, drop_prob: float) -> None:
        r'''Initialization of decoder.

            Args:
                pe_scheme (str): Position embedding scheme (range: ['nope', 'ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire', 'cho']).
                emb_dim (int): Embedding dimension.
                num_heads (int): The number of heads used in multi-head attention.
                num_ffn_layers (int): The number of layers in feed forward module.
                hidden_dim (int): Hidden dimension in feed forward module.
                activate_fn (str): Activation function (range: ['relu', 'gelu', 'silu']).
                norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
                drop_prob (float): Dropout probability.
        '''
        super().__init__()
        self.pe_scheme = pe_scheme
        if pe_scheme == 'rpe-s':
            self.mha = TransformerXLAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)
        elif pe_scheme == 'rpe-l':
            self.mha = DisentangledAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)
        else:
            self.mha = UniversalAttention(
                emb_dim, num_heads, norm_scheme, drop_prob)
        self.ffn = FeedForward(num_ffn_layers, emb_dim,
                               hidden_dim, activate_fn, norm_scheme, drop_prob)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], rpe: Optional[Union[Tensor, Tuple[Tensor]]], b: Optional[Tensor], ra: Optional[Tensor], teacher_forcing: bool, refresh_cache: bool, max_seq_len: Optional[int]) -> Tensor:
        r'''Forward process of a decoder layer in decoder-only Transformer.

        Args:
            x (Tensor): Embeddings of sequences or current tokens.
            key_padding_mask (Optional[Tensor]): Key padding mask for sequences or current tokens.
            attn_mask (Optional[Tensor]): Attention mask.
            rpe (Optional[Union[Tensor, Tuple[Tensor]]]): Relative position embeddings.
            b (Optional[Tensor]): Relative bias.
            ra (Optional[Tensor]): Rotation angles for Q & K.
            teacher_forcing (bool): Whether conducts teacher forcing training or evaluation.
            refresh_cache (bool): Whether refreshes K & V cache slots (and resets hit positions in (Qr &) Kr cache slots).
            max_seq_len (Optional[int]): The maximum length of a full sequence.

        Shapes:
            Input:
                Embeddings of sequences or current tokens
                    Training: `(B, N_p + N_r, D)`
                        `B`: the number of sequences
                        `N_p`: the length of a prompt
                        `N_r`: the length of a full response
                    Inference:
                        2.1 Beginning: `(B, N_p + 1, D)`
                        2.2 Later: `(B, D)`
                Key padding mask for sequences or current tokens
                    Training: `(B, L)`
                    Inference: `(B, N_p + L_r)`
                        `L_r`: the length of a current response
                Attention mask
                    Training:
                        `(N_p + N_r, N_p + N_r)`
                    Inference:
                        2.1 Beginning: `(N_p + 1, N_p + 1)`
                        2.2 Later: `None`
                Relative position embeddings (RPE-S, RPE-L)
                    RPE-S:
                        1.1 Training:
                            `(2 * (N_p + N_r) - 1, D)`
                        1.2 Inference:
                            1.2.1 Batch 0:
                                1.2.1.1 Beginning: `(2* (N0_p + 1) - 1, D)`
                                    `N0_p`: the length of a prompt in batch 0
                                1.2.1.2 Later: `(D)`
                            1.2.2 Batch b (b > 0):
                                1.2.2.1 Beginning:
                                    1.2.2.1.1 If Nb_p + 1 > NRC: `((Nb_p - NLC + 1, D), (Nb_p - NRC + 1, D))`
                                        `Nb_p`: the length of a prompt in batch b
                                        `NRC`: the number of cached positions above zero
                                        `NLC`: the number of cached positions below zero
                                    1.2.2.1.2 If NLC < Nb_p + 1 <= NRC: `(None, (Nb_p - NRC + 1, D))`
                                    1.2.2.1.3 Else: `None`
                                1.2.2.2 Later:
                                    1.2.2.2.1 If Nb_p + 1 + nb > NRC: `(D)`
                                        `nb`: step nb (0 < nb < Nb_r)
                                        `Nb_r`: the length of a full response in batch b
                                    1.2.2.2.2 Else: `None`
                    RPE-L:
                        2.1 Training:
                            Same as RPE-S
                        2.2 Inference:
                            2.2.1 Batch 0:
                                2.2.1.1 Beginning: Same as RPE-S
                                2.2.1.2 Later: `((D), (D))`
                            2.2.2 Batch b (b > 0):
                                2.2.2.1 Beginning:
                                    Same as RPE-S
                                2.2.2.2 Later:
                                    1.2.2.2.1 If Nb_p + 1 + nb > NRC: `((D), (D))`
                                    1.2.2.2.2 Else: `None`
                Relativa bias (T5, ALiBi, FIRE)
                    Training: `(H, N_p + N_r, N_p + N_r)`
                        `H`: the number of heads
                    Inference:
                        2.1 Beginning: `(H, N_p + 1)`
                        2.2 Later: `(B, H, N_p + L_r)`
                Rotation angles for Q & K (RoPE):
                    Training:
                        `(N_p + N_r, D / 2)`
                    Inference:
                        2.1 Beginning: `(N_p + 1, D / 2)`
                        2.2 Later: `(B, D / 2)`
            Output:
                Embeddings of output sequences or next tokens
                    Training:
                        `(B, N_p + N_r, D)`
                    Inferencee:
                        2.1 Beginning: `(B, N_p + 1, D)`
                        2.2 Later: `(B, D)`
        '''
        requires_cache = not teacher_forcing
        if self.pe_scheme in ['rpe-s', 'rpe-l']:
            x = self.mha(x, x, key_padding_mask, attn_mask, rpe,
                         teacher_forcing, requires_cache, max_seq_len, refresh_cache)
        else:
            x = self.mha(x, x, key_padding_mask, attn_mask, b, ra,
                         requires_cache, refresh_cache, max_seq_len)
        x = self.ffn(x)

        return x


class IndexHelp:
    r'''Class for indexing for cache accesses.'''

    def __init__(self) -> None:
        r'''Initialization of class.'''
        # Index of current token in an output/response sequence
        self.step = 0
        # Boundaries for (Qr &) Kr caches in former batch(es)
        self.lbound = self.rbound = 0
        # Maximum index in an input sequence (encoder-decoder) in former batch(es)
        self.in_bound = 0
        # Maximum index in an output sequence (encoder-decoder) in former batch(es)
        self.out_bound = -1

    def reset_step(self) -> None:
        r'''Reset step at the beginning of inference in a batch.'''
        self.step = 0


class Transformer(Module):
    r'''Transformer models designed for learning polynomial arithmetic.'''

    def __init__(self, pe_scheme: str, vocab_size: int, emb_dim: int, num_heads: int, num_ffn_layers: int, hidden_dim: int, activate_fn: str, norm_scheme: str, drop_prob: float, num_encoder_layers: int, num_decoder_layers: int) -> None:
        r'''Initialization of module.

        Args:
            pe_scheme (str): Position embedding scheme (range: ['nope', 'ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire', 'pc']).
            vocab_size (int): Size of vocabulary.
            emb_dim (int): Embedding dimension.
            num_heads (int): The number of heads used in multi-head attention.
            num_ffn_layers (int): The number of layers in feed forward module.
            hidden_dim (int): Hidden dimension in feed forward module.
            activate_fn (str): Activation function (range: ['relu', 'gelu', 'silu']).
            norm_scheme (str): Normalization scheme (range: ['layer', 'rms']).
            drop_prob (float): Dropout probability.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
        '''
        super().__init__()
        self.pe_scheme = pe_scheme
        self.decoder_only = num_encoder_layers == 0
        self.word_emb = WordEmbedding(vocab_size, emb_dim)
        if pe_scheme in ['t5', 'alibi', 'fire']:
            self.pos_emb = PositionEmbedding(
                emb_dim, pe_scheme, self.decoder_only, num_heads)
        else:
            self.pos_emb = PositionEmbedding(
                emb_dim, pe_scheme, self.decoder_only)
        # Decoder-only architecture if no encoder layer
        if self.decoder_only:
            self.decoder = ModuleList([TransformerDecoderOnlyLayer(pe_scheme, emb_dim, num_heads, num_ffn_layers,
                                      hidden_dim, activate_fn, norm_scheme, drop_prob) for _ in range(num_decoder_layers)])
        else:
            self.encoder = ModuleList([TransformerEncoderLayer(pe_scheme, emb_dim, num_heads, num_ffn_layers,
                                      hidden_dim, activate_fn, norm_scheme, drop_prob) for _ in range(num_encoder_layers)])
            self.encoder_norm = LayerNorm(
                emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
            self.decoder = ModuleList([TransformerDecoderLayer(pe_scheme, emb_dim, num_heads, num_ffn_layers,
                                      hidden_dim, activate_fn, norm_scheme, drop_prob) for _ in range(num_decoder_layers)])
        self.decoder_norm = LayerNorm(
            emb_dim) if norm_scheme == 'layer' else RMSNorm(emb_dim)
        self.out_proj = Linear(emb_dim, vocab_size, bias=False)
        self.out_proj.weight = self.word_emb.emb.weight
        self.offset = SEQ_LEN - 1

    def forward(self, x_out: Optional[Tensor] = None, out_mask: Optional[Tensor] = None, x_in: Optional[Tensor] = None, in_mask: Optional[Tensor] = None, x: Optional[Tensor] = None, x_mask: Optional[Tensor] = None, posid: Optional[Tensor] = None, posid1: Optional[Tensor] = None, posid2: Optional[Tensor] = None, max_seq_len: Optional[Union[int, List]] = None, teacher_forcing: bool = False) -> Tensor:
        r'''Forward process of Transformer models.

        Args:
            x_out (Optional[Tensor]): Vocabulary indices of tokens in output sequences (default: None).
            out_mask (Optional[Tensor]): Key padding mask for output sequences (default: None).
            x_in (Optional[Tensor]): Vocabulary indices of tokens in input sequences (default: None).
            in_mask (Optional[Tensor]): Key padding mask for input sequences (default: None).
            x (Optional[Tensor]): Vocabulary indices of tokens in sequences (default: None).
            x_mask (Optional[Tensor]): Key padding mask for sequences (default: None).
            posid (Optional[Tensor]): Position IDs for current tokens in responses (default: None).
            posid1 (Optional[Tensor]): Position IDs for digit coupling (default: None).
            posid2 (Optional[Tensor]): Position IDs for operand coupling (default: None).
            max_seq_len (Optional[Union[int, List]]): Maximum length of an output sequence (encoder-decoder) or a full sequence (decoder-only), or maximum lengths of an input sequence and an output sequence (decoder-only when `pe_scheme` in ['RPE-S', 'RPE-L']) (default: None).
            teacher_forcing (bool): Whether conducts teacher forcing training or evaluation (default: False).

        Shape:
            Input:
                Vocabulary indices of tokens in output sequences (encoder-decoder)
                    Training: `(B, M)`
                        `B`: the number of sequences
                        `M`: the length of a full output sequence
                    Inference: `(B)`
                Key padding mask for output sequences (encoder-decoder)
                    Training: `(B, M)`
                    Inference: `None`
                Vocabulary indices of tokens in input sequences (encoder-decoder)
                    Training: `(B, N)`
                        `N`: the length of an input sequence
                    Inference:
                        2.1 Beginning: `(B, N)`
                        2.2 Later: `None`
                Key padding mask for input sequences (encoder-decoder)
                    Training/Batched inference: `(B, N)`
                    Single inference: `None`
                Vocabulary indices of tokens in sequences (decoder-only)
                    Training: `(B, N_p + N_r)`
                        `N_p`: the length of a prompt
                        `N_r`: the length of a full response
                    Inference:
                        2.1 Beginning: `(B, N_p + 1)`
                        2.2 Later: `(B)`
                Key padding mask for sequences (decoder-only)
                    Training: `(B, N_p + N_r)`
                    Inference: `(B, N_p + L_r)`
                        `L_r`: the length of a current response
                Position IDs for current tokens in responses (decoder-only)
                    Training: `None`
                    Infernece:
                        2.1 Beginning:
                            `None`
                        2.2 Later:
                            2.2.1 T5/ALiBi/FIRE: `(B, N_p + L_r)`
                            2.2.2 Other PE: `None`
                Position IDs for digit coupling (decoder-only)
                    Same as 'vocabulary indices of tokens in sequences (decoder-only)'
                Position IDs for operand coupling (decoder-only)
                    Same as 'vocabulary indices of tokens in sequences (decoder-only)'
            Output:
                Embeddings of sequences or next tokens
                    Training:
                        1.1 Encoder-decoder: `(B, M, D)`
                        1.2 Decoder-only: `(B, N_p + N_r, D)`
                    Inference: `(B, D)`
        '''
        # Whether refreshes K & V cache slots (and resets hit positions in (Qr &) Kr cache slots)
        refresh_cache = False
        if not teacher_forcing and self.decoder_only and x.dim() == 2:
            refresh_cache = True
        elif not teacher_forcing and x_in != None:
            refresh_cache = True

        # Word embedding
        if self.decoder_only:
            x = self.word_emb(x)
        else:
            x_out = self.word_emb(x_out)
            if x_in != None:
                x_in = self.word_emb(x_in)

        # Index assisting cache hits & updates
        if not teacher_forcing and not hasattr(self, 'index'):
            self.index = IndexHelp()
        if refresh_cache:
            self.index.reset_step()

        # Position embedding, relative bias or rotation angles
        if self.pe_scheme in ['nope', 'ape-s', 'ape-l', 'pc']:
            if self.decoder_only:
                rpe = b = ra = None
                if self.pe_scheme in ['ape-s', 'ape-l']:
                    if teacher_forcing:
                        x += self.pos_emb.ape[:x.size(-2)]
                    elif refresh_cache:
                        x += self.pos_emb.ape[:x.size(-2)]
                        self.index.step += x.size(-2)
                    else:
                        if posid is None:
                            x += self.pos_emb.ape[posid]
                        else:
                            x += self.pos_emb.ape[:x.size(-2)]
                        self.index.step += 1
                elif self.pe_scheme == 'pc':
                    x += self.pos_emb.id1_pe(posid1) + \
                        self.pos_emb.id2_pe(posid2)
            else:
                if x_in != None:
                    rpe_in = b_in = ra_in = None
                rpe_out = b_out = ra_out = None
                if self.pe_scheme == 'ape-s':
                    if teacher_forcing:
                        x_out += self.pos_emb.ape[:x_out.size(-2)]
                        x_in += self.pos_emb.ape[:x_in.size(-2)]
                    else:
                        x_out += self.pos_emb.ape[self.index.step]
                        self.index.step += 1
                        if refresh_cache:
                            x_in += self.pos_emb.ape[:x_in.size(-2)]
                elif self.pe_scheme == 'ape-l':
                    if teacher_forcing:
                        x_out += self.pos_emb.ape_out[:x_out.size(-2)]
                        x_in += self.pos_emb.ape_in[:x_in.size(-2)]
                    else:
                        x_out += self.pos_emb.ape_out[self.index.step]
                        self.index.step += 1
                        if refresh_cache:
                            x_in += self.pos_emb.ape_in[:x_in.size(-2)]
        elif self.pe_scheme in ['rpe-s', 'rpe-l']:
            if self.decoder_only:
                b = ra = None
                if teacher_forcing:
                    rpe = self.pos_emb.rpe[-x.size(-2) + 1 +
                                           self.offset:x.size(-2) + self.offset]
                elif refresh_cache:
                    n = x.size(-2)
                    if self.index.rbound:
                        newleft = newright = None
                        if -n < self.index.lbound:
                            newleft = self.pos_emb.rpe[-n + 1 +
                                                       self.offset:self.index.lbound + self.offset]
                            self.lbound = -n + 1
                        if n > self.index.rbound:
                            newright = self.pos_emb.rpe[self.index.rbound +
                                                        1 + self.offset:n + self.offset]
                            self.rbound = n - 1
                        rpe = (newleft, newright)
                    else:
                        rpe = self.pos_emb.rpe[-n + 1 +
                                               self.offset:n + self.offset]
                        self.lbound = -n + 1
                        self.rbound = n - 1
                    self.index.step += n
                else:
                    if self.index.step > self.index.bound:
                        if self.pe_scheme == 'rpe-s':
                            rpe = self.pos_emb.rpe[self.index.step + self.offset]
                        else:
                            rpe = (self.pos_emb.rpe[self.index.step + self.offset],
                                   self.pos_emb.rpe[-self.index.step + self.offset])
                        self.index.bound += 1
                    else:
                        rpe = None
                    self.index.step += 1
            else:
                if x_in != None:
                    b_in = ra_in = None
                b_out = ra_out = None
                if teacher_forcing:
                    if self.pe_scheme == 'rpe-s':
                        rpe_out = self.pos_emb.rpe[-x_out.size(-2) +
                                                   1 + self.offset:x_out.size(-2) + self.offset]
                        rpe_in = self.pos_emb.rpe[-x_in.size(-2) +
                                                  1 + self.offset:x_in.size(-2) + self.offset]
                    else:
                        rpe_out = self.pos_emb.rpe_out[-x_out.size(-2) +
                                                       1 + self.offset:x_out.size(-2) + self.offset]
                        rpe_in = self.pos_emb.rpe_in[-x_in.size(-2) +
                                                     1 + self.offset:x_in.size(-2) + self.offset]
                else:
                    if refresh_cache:
                        n = x_in.size(-2)
                        if n > self.index.in_bound:
                            if self.pe_scheme == 'rpe-s':
                                if self.index.in_bound:
                                    rpe_in = (self.pos_emb.rpe[-n + 1 + self.offset:-self.index.in_bound + 1 +
                                              self.offset], self.pos_emb.rpe[self.index.in_bound + self.offset:n+self.offset])
                                else:
                                    rpe_in = self.pos_emb.rpe[-n +
                                                              1 + self.offset:n + self.offset]
                            else:
                                if self.index.in_bound:
                                    rpe_in = (self.pos_emb.rpe_in[-n + 1 + self.offset:-self.index.in_bound + 1 +
                                              self.offset], self.pos_emb.rpe_in[self.index.in_bound + self.offset:n+self.offset])
                                else:
                                    rpe_in = self.pos_emb.rpe_in[-n +
                                                                 1 + self.offset:n + self.offset]
                            self.index.in_bound = n
                        else:
                            rpe_in = None
                    if self.index.step > self.index.out_bound:
                        if self.pe_scheme == 'rpe-s':
                            rpe_out = self.pos_emb.rpe[self.index.step +
                                                       self.offset]
                        else:
                            rpe_out = (self.pos_emb.rpe_out[self.index.step + self.offset],
                                       self.pos_emb.rpe_out[-self.index.step + self.offset])
                        self.index.out_bound += 1
                    else:
                        rpe_out = None
                    self.index.step += 1
        elif self.pe_scheme == 'rope':
            if self.decoder_only:
                rpe = b = None
                if teacher_forcing:
                    ra = self.pos_emb.ra[:x.size(-2)]
                elif refresh_cache:
                    ra = self.pos_emb.ra[:x.size(-2)]
                    self.index.step += x.size(-2)
                else:
                    ra = self.pos_emb.ra[self.index.step]
                    self.index.step += 1
            else:
                if x_in != None:
                    rpe_in = b_in = None
                rpe_out = b_out = None
                if teacher_forcing:
                    ra_out = self.pos_emb.ra[:x_out.size(-2)]
                    ra_in = self.pos_emb.ra[:x_in.size(-2)]
                else:
                    ra_out = self.pos_emb.ra[self.index.step]
                    self.index.step += 1
                    if refresh_cache:
                        ra_in = self.pos_emb.ra[:x_in.size(-2)]
        else:
            if self.decoder_only:
                rpe = ra = None
                if teacher_forcing:
                    b = self.pos_emb.b[:, :x.size(-2), :x.size(-2)]
                elif refresh_cache:
                    b = self.pos_emb.b[:, x.size(-2) - 1, :x.size(-2)]
                    self.index.step += x.size(-2)
                else:
                    b = self.pos_emb.b.unsqueeze(0).expand(posid.size(0), -1, -1, -1).gather(
                        2, posid[:, None, :, None].expand(-1, posid.size(0), -1, posid.size(2))).gather(
                        3, posid[:, None, None, :].expand(-1, posid.size(0), posid.size(1), -1))
                    self.index.step += 1
            else:
                if x_in != None:
                    rpe_in = ra_in = None
                rpe_out = ra_out = None
                if teacher_forcing:
                    if self.pe_scheme == 'alibi':
                        b_out = self.pos_emb.b[:,
                                               :x_out.size(-2), :x_out.size(-2)]
                        b_in = self.pos_emb.b[:,
                                              :x_in.size(-2), :x_in.size(-2)]
                    else:
                        b_out = self.pos_emb.b_out[:,
                                                   :x_out.size(-2), :x_out.size(-2)]
                        b_in = self.pos_emb.b_in[:,
                                                 :x_in.size(-2), :x_in.size(-2)]
                else:
                    if self.pe_scheme == 'alibi':
                        b_out = self.pos_emb.b[:,
                                               self.index.step, :self.index.step + 1]
                    else:
                        b_out = self.pos_emb.b_out[:,
                                                   self.index.step, :self.index.step + 1]
                    self.index.step += 1
                    if refresh_cache:
                        if self.pe_scheme == 'alibi':
                            b_in = self.pos_emb.b[:,
                                                  :x_in.size(-2), :x_in.size(-2)]
                        else:
                            b_in = self.pos_emb.b_in[:,
                                                     :x_in.size(-2), :x_in.size(-2)]

        # Attention mask
        if self.decoder_only and (teacher_forcing or refresh_cache):
            attn_mask = triu(ones(x.size(-2), x.size(-2)),
                             diagonal=1).bool().to(next(self.parameters()).device)
        elif teacher_forcing:
            attn_mask = triu(ones(x_out.size(-2), x_out.size(-2)),
                             diagonal=1).bool().to(next(self.parameters()).device)
        else:
            attn_mask = None

        if self.decoder_only:
            # Decoder-only Transformer
            for layer in self.decoder:
                x = layer(x, x_mask, attn_mask, rpe, b, ra,
                          teacher_forcing, refresh_cache, max_seq_len)
            if refresh_cache:
                x = self.decoder_norm(x[-1] if x.dim() == 2 else x[:, -1])
            else:
                # Training, later inference steps
                x = self.decoder_norm(x)
            return self.out_proj(x)
        else:
            # Encoder-decoder Transformer
            if x_in != None:
                for layer in self.encoder:
                    x_in = layer(x_in, in_mask, rpe_in, b_in, ra_in)
            for layer in self.decoder:
                x_out = layer(x_out, out_mask, x_in, in_mask,
                              attn_mask, rpe_out, b_out, ra_out, teacher_forcing, refresh_cache, max_seq_len)
            return self.out_proj(x_out)

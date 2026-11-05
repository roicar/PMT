from torch.nn import Module, Embedding, Parameter
from torch import empty, Tensor, matmul, zeros
from torch.nn.init import normal_
from torch_geometric.nn import GraphConv
from torch_geometric.data import HeteroData
from utils import copy_dict_to_device
from torch.nn.utils.rnn import pad_sequence
from model_utils.transformer import TransformerEncoderLayer, TransformerDecoderLayer
from typing import Final
from typing import Final, Optional, Union, Tuple, List
from torch.nn import Module, LayerNorm, ReLU, GELU, Linear, Dropout, ModuleList
from model_utils.norm import RMSNorm
from model_utils.activation import SiLU
from torch import Tensor, triu, ones
from model_utils.attention import TransformerXLAttention, DisentangledAttention, UniversalAttention
from model_utils.embedding import WordEmbedding, PositionEmbedding

# Maximum number of tokens in a sequence
SEQ_LEN: Final = 768
MAX_NUM_DIGITS: Final = 32

class GraphEmbedding(Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        self.emb_dim = emb_dim
        # token word embeddings
        self.word_emb = Embedding(vocab_size, emb_dim)
        # Digitwise weights for coefficients and exponents
        self.w_coeff = Parameter(empty(MAX_NUM_DIGITS + 1, emb_dim, emb_dim))
        self.w_exp = Parameter(empty(MAX_NUM_DIGITS, emb_dim, emb_dim))
        normal_(self.w_coeff)
        normal_(self.w_exp)
        # Graph convolutional layers for other edges
        self.pow_conv = GraphConv(-1, emb_dim, bias=False)
        self.pow_conv.lin_root.weight = self.pow_conv.lin_rel.weight
        self.term_conv = GraphConv(-1, emb_dim, bias=False)
        self.term_conv.lin_root.weight = self.term_conv.lin_rel.weight
        self.poly_conv = GraphConv(-1, emb_dim, bias=False)
        self.poly_conv.lin_root.weight = self.poly_conv.lin_rel.weight
        
    def digitwise_aggregate(self, src_emb: Tensor, edge_index: Tensor, dst_emb: Tensor, w_pos: Parameter, digit_type: str = 'coeff', backward: bool = False, reverse: bool = False):
        src, dst = edge_index
        if backward:
            for idx in range(src_emb.size(0)):
                digits = dst[src == idx]
                num_digits = len(digits)
                if digit_type == 'coeff':
                    if reverse:
                        positions = list(range(1, num_digits)) + [0]
                    else:
                        positions = [0] + list(range(num_digits - 1, 0, -1))
                else:  # 'exp'
                    if reverse:
                        positions = list(range(num_digits))
                    else:
                        positions = list(range(num_digits - 1, -1, -1))
                for i, pos in enumerate(positions):
                    dst_emb[digits[i]
                            ] += matmul(src_emb[idx], w_pos[pos].transpose(0, 1))
        else:
            for idx in range(dst_emb.size(0)):
                digits = src[dst == idx]
                num_digits = len(digits)
                if digit_type == 'coeff':
                    if reverse:
                        positions = list(range(1, num_digits)) + [0]
                    else:
                        positions = [0] + list(range(num_digits - 1, 0, -1))
                else:
                    if reverse:
                        positions = list(range(num_digits))
                    else:
                        positions = list(range(num_digits - 1, -1, -1))
                for i, pos in enumerate(positions):
                    dst_emb[digits[i]
                            ] += matmul(src_emb[idx], w_pos[pos].transpose(0, 1))
        return dst_emb
    
    def forward(self, graphs:HeteroData):
        device = next(self.parameters()).device
        x_dict = {
            'token': self.word_emb(graphs['token'].x.squeeze(-1).to(device)),
            'coeff': graphs.zeros(graphs['coeff'].x.size(0), self.emb_dim).to(device),
            'exp': zeros(graphs['exp'].x.size(0), self.emb_dim).to(device),
            'pow': zeros(graphs['pow'].x.size(0), self.emb_dim).to(device),
            'term': zeros(graphs['term'].x.size(0), self.emb_dim).to(device),
            'poly': zeros(graphs['poly'].x.size(0), self.emb_dim).to(device)
        }
        edge_dict = copy_dict_to_device(graphs.edge_index_dict, device)
        ptr = graphs['token'].ptr.to(device)
        # Forward
        x_dict['coeff'] = self.digitwise_aggregate(x_dict['token'], edge_dict[(
            'token', 'to', 'coeff')], x_dict['coeff'], self.w_coeff)
        x_dict['exp'] = self.digitwise_aggregate(x_dict['token'], edge_dict[(
            'token', 'to', 'exp')], x_dict['exp'], self.w_exp, 'exp')
        x_dict['pow'] = self.pow_conv(
            (x_dict['token'], x_dict['pow']), edge_dict[('token', 'to', 'pow')])
        x_dict['pow'] = self.pow_conv(
            (x_dict['exp'], x_dict['pow']), edge_dict[('exp', 'to', 'pow')])
        x_dict['term'] = self.term_conv(
            (x_dict['coeff'], x_dict['term']), edge_dict[('coeff', 'to', 'term')])
        x_dict['term'] = self.term_conv(
            (x_dict['pow'], x_dict['term']), edge_dict[('pow', 'to', 'term')])
        x_dict['poly'] = self.poly_conv(
            (x_dict['term'], x_dict['poly']), edge_dict[('term', 'to', 'poly')])
        # Backward
        x_dict['term'] = self.poly_conv(
            (x_dict['poly'], x_dict['term']), edge_dict[('poly', 'to', 'term')])
        x_dict['pow'] = self.term_conv(
            (x_dict['term'], x_dict['pow']), edge_dict[('term', 'to', 'pow')])
        x_dict['coeff'] = self.term_conv(
            (x_dict['term'], x_dict['coeff']), edge_dict[('term', 'to', 'coeff')])
        x_dict['exp'] = self.pow_conv(
            (x_dict['pow'], x_dict['exp']), edge_dict[('pow', 'to', 'exp')])
        x_dict['token'] = self.pow_conv(
            (x_dict['pow'], x_dict['token']), edge_dict[('pow', 'to', 'token')])
        x_dict['token'] = self.digitwise_aggregate(x_dict['exp'], edge_dict[(
            'exp', 'to', 'token')], x_dict['token'], self.w_exp, 'exp', True)
        x_dict['token'] = self.digitwise_aggregate(x_dict['coeff'], edge_dict[(
            'coeff', 'to', 'token')], x_dict['token'], self.w_coeff, 'coeff', True)

        emb = pad_sequence([x_dict['token'][ptr[i]:ptr[i + 1]]
                           for i in range(len(ptr) - 1)], batch_first=True)
        mask = emb.abs().sum(dim=-1) == 0
        return emb, mask
    
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


class Polynormer(Module):
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
        self.graphemb = GraphEmbedding(vocab_size, emb_dim)
        self.word_emb = WordEmbedding(vocab_size, emb_dim)
        if pe_scheme in ['t5', 'alibi', 'fire']:
            self.pos_emb = PositionEmbedding(
                emb_dim, pe_scheme, self.decoder_only, num_heads)
        else:
            self.pos_emb = PositionEmbedding(
                emb_dim, pe_scheme, self.decoder_only)
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
        
    def forward(self, x_out: Optional[Tensor] = None, out_mask: Optional[Tensor] = None, x_in: Optional[HeteroData]=None, max_seq_len: Optional[Union[int, List]] = None, teacher_forcing: bool = False) -> Tensor:
        # Whether refreshes K & V cache slots (and resets hit positions in (Qr &) Kr cache slots)
        refresh_cache = False
        if not teacher_forcing and x_in != None:
            refresh_cache = True

        # Word embedding
        x_out = self.word_emb(x_out)

        # Index assisting cache hits & updates
        if not teacher_forcing and not hasattr(self, 'index'):
            self.index = IndexHelp()
        if refresh_cache:
            self.index.reset_step()
        
        if x_in != None:
            rpe_in = b_in = ra_in = None
            rpe_out = b_out = ra_out = None
        if teacher_forcing:
            x_out += self.pos_emb.ape_out[:x_out.size(-2)]
            x_in, mask = self.graphemb(x_in)
            x_in += self.pos_emb.ape_in[:x_in.size(-2)]
        else:
            x_out += self.pos_emb.ape_out[self.index.step]
            self.index.step += 1
            if refresh_cache:
                x_in, mask = self.graphemb(x_in)
        if teacher_forcing:
            attn_mask = triu(ones(x_out.size(-2), x_out.size(-2)),
                             diagonal=1).bool().to(next(self.parameters()).device)
        else:
            attn_mask = None
            # Encoder-decoder Transformer
            if x_in != None:
                for layer in self.encoder:
                    x_in = layer(x_in, x_in, rpe_in, b_in, ra_in)
            for layer in self.decoder:
                x_out = layer(x_out, out_mask, x_in, x_in,
                              attn_mask, rpe_out, b_out, ra_out, teacher_forcing, refresh_cache, max_seq_len)
            if refresh_cache:
                return self.out_proj(x_out), mask
            return self.out_proj(x_out)
        
        
        
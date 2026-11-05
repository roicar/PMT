'''
    Utilities for arguments\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 19 Feb 2026
'''
from argparse import Namespace, ArgumentParser
from torch import __version__
from re import match
from os.path import join, abspath, dirname


def check_model_params(args: Namespace, param_suffix: str = '', print_info: str = '') -> None:
    r'''Check whether values of model parameters are legal.'''
    if param_suffix == '_accum':
        if args.pe_scheme_accum not in ['ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire', 'pc']:
            raise ValueError('Position embedding scheme' + print_info +
                             ' can only RANGE AMONG [\'ape-s\', \'ape-l\', \'rpe-s\', \'rpe-l\', \'t5\', \'alibi\', \'rope\', \'fire\', \'pc\']!')
    elif eval('args.pe_scheme' + param_suffix) not in ['ape-s', 'ape-l', 'rpe-s', 'rpe-l', 't5', 'alibi', 'rope', 'fire']:
        raise ValueError('Position embedding scheme' + print_info +
                         ' can only RANGE AMONG [\'ape-s\', \'ape-l\', \'rpe-s\', \'rpe-l\', \'t5\', \'alibi\', \'rope\', \'fire\']!')
    if eval('args.emb_dim' + param_suffix) <= 0:
        raise ValueError('Embedding dimension' +
                         print_info + ' should be POSITIVE!')
    if eval('args.num_heads' + param_suffix) <= 0:
        raise ValueError('The number of heads in multi-head attention' +
                         print_info + ' should be POSITIVE!')
    if eval('args.emb_dim' + param_suffix) % eval('args.num_heads' + param_suffix) != 0 or eval('args.emb_dim' + param_suffix) % 2 != 0:
        raise ValueError('The embedding dimension' + print_info +
                         ' must be DIVISIBLE BY the number of heads and EVEN!')
    if eval('args.num_ffn_layers' + param_suffix) < 2:
        raise ValueError('The number of feed forward layers' +
                         print_info + ' should be NO LESS THAN 2!')
    if eval('args.hidden_dim' + param_suffix) <= eval('args.emb_dim' + param_suffix):
        raise ValueError('The hidden dimension in feed forward network' +
                         print_info + ' should be GREATER THAN the embedding dimension!')
    if eval('args.activate_fn' + param_suffix) not in ['relu', 'gelu', 'silu']:
        raise ValueError('Activation function' + print_info +
                         ' can only RANGE AMONG [\'relu\', \'gelu\', \'silu\']!')
    if eval('args.norm_scheme' + param_suffix) not in ['layer', 'rms']:
        raise ValueError('Normalization scheme' + print_info +
                         'can only RANGE AMONG [\'layer\', \'rms\']!')
    if param_suffix != '_accum':
        if eval('args.num_encoder_layers' + param_suffix) <= 0:
            raise ValueError('The number of encoder layers' +
                             print_info + ' should be POSITIVE!')
    if eval('args.num_decoder_layers' + param_suffix) <= 0:
        raise ValueError('The number of decoder layers' +
                         print_info + ' should be POSITIVE!')


def assign_arguments() -> Namespace:
    r'''Assign arguments for data processing, model training and inference.'''
    '''
        Extract arguments from input `"args"`
    '''
    parser = ArgumentParser()

    # Supplementary experiment only: representation and explicit checkpoint paths.
    parser.add_argument('--unsigned', action='store_true',
                        help='Use the unsigned scratchpad and PosIDs shown in the comparison figure')
    parser.add_argument('--accum_checkpoint', type=str, default='')
    parser.add_argument('--collect_checkpoint', type=str, default='')
    # Universal settings
    parser.add_argument('--task', type=str, default='mul')
    parser.add_argument('--dataset', type=str, default='v1d1C1')
    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tokenization', type=str, default='ExDs',
                        help='tokenization scheme')
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--log', action='store_true')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='the maximum number of epochs to wait before early stopping')
    parser.add_argument('--lod', type=float, default=0,
                        help='limit of detection in accuracy increase')

    # Normal model
    parser.add_argument('--pe_scheme', type=str, default='ape-l',
                        help='position embedding scheme')
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_ffn_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--activate_fn', type=str, default='relu')
    parser.add_argument('--norm_scheme', type=str, default='layer')
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)

    # Multiplication subtask 1: learning collected product
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--pe_scheme_collect', type=str, default='ape-l')
    parser.add_argument('--emb_dim_collect', type=int, default=512)
    parser.add_argument('--num_heads_collect', type=int, default=8)
    parser.add_argument('--num_ffn_layers_collect', type=int, default=2)
    parser.add_argument('--hidden_dim_collect', type=int, default=2048)
    parser.add_argument('--activate_fn_collect', type=str, default='relu')
    parser.add_argument('--norm_scheme_collect', type=str, default='layer')
    parser.add_argument('--num_encoder_layers_collect', type=int, default=6)
    parser.add_argument('--num_decoder_layers_collect', type=int, default=6)
    # Multiplication subtask 2a/3a: combination of collected product
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--pe_scheme_combine', type=str, default='ape-l')
    parser.add_argument('--emb_dim_combine', type=int, default=512)
    parser.add_argument('--num_heads_combine', type=int, default=8)
    parser.add_argument('--num_ffn_layers_combine', type=int, default=2)
    parser.add_argument('--hidden_dim_combine', type=int, default=2048)
    parser.add_argument('--activate_fn_combine', type=str, default='relu')
    parser.add_argument('--norm_scheme_combine', type=str, default='layer')
    parser.add_argument('--num_encoder_layers_combine', type=int, default=6)
    parser.add_argument('--num_decoder_layers_combine', type=int, default=6)
    # Integer accumulation, multiplication subtask 2b / addition subtask 2
    parser.add_argument('--accum', action='store_true')
    parser.add_argument('--pe_scheme_accum', type=str, default='pc')
    parser.add_argument('--emb_dim_accum', type=int, default=512)
    parser.add_argument('--num_heads_accum', type=int, default=8)
    parser.add_argument('--num_ffn_layers_accum', type=int, default=2)
    parser.add_argument('--hidden_dim_accum', type=int, default=2048)
    parser.add_argument('--activate_fn_accum', type=str, default='relu')
    parser.add_argument('--norm_scheme_accum', type=str, default='layer')
    parser.add_argument('--num_decoder_layers_accum', type=int, default=6)
    # Long multiplication subtask 3b / Addition subtask 1: learning collected sum of combined subproducts
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--pe_scheme_sum', type=str, default='ape-l')
    parser.add_argument('--emb_dim_sum', type=int, default=512)
    parser.add_argument('--num_heads_sum', type=int, default=8)
    parser.add_argument('--num_ffn_layers_sum', type=int, default=2)
    parser.add_argument('--hidden_dim_sum', type=int, default=2048)
    parser.add_argument('--activate_fn_sum', type=str, default='relu')
    parser.add_argument('--norm_scheme_sum', type=str, default='layer')
    parser.add_argument('--num_encoder_layers_sum', type=int, default=6)
    parser.add_argument('--num_decoder_layers_sum', type=int, default=6)

    # Pseudodivision
    parser.add_argument('--prem_step', type=str, default='0',
                        help='the number of pseudodivision steps \'seen\' by the model (default: \'0\')')
    parser.add_argument('--detailed_spa', action='store_true',
                        help='whether to exploit detailed scratchpad during pseudodivision (default: False)')

    # Finetune
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no_freeze', action='store_true')
    args = parser.parse_args()

    '''
        Check whether values of parameters are legal
    '''
    # General check
    if args.task not in ['mul', 'add', 'prem', 'charset']:
        raise ValueError(
            'Learning task can only RANGE AMONG [\'mul\', \'add\', \'prem\', \'charset\']!')
    if args.train and args.valid:
        raise ValueError(
            'Whether to conduct training or inference should be specified!')
    if args.batch_size < 0:
        raise ValueError('Batch size sould be NONNEGATIVE!')
    if args.tokenization not in ['ImUs', 'ImDs', 'ExUs', 'ExDs']:
        raise ValueError(
            'Tokenization scheme can only RANGE AMONG [\'ImUs\', \'ImDs\', \'ExUs\', \'ExDs\']!')
    if args.model not in ['transformer', 'polynormer']:
        raise ValueError(
            'Model architecture can only RANGE AMONG [\'transformer\', \'polynormer\']!')
    if args.drop_prob < 0 or args.drop_prob >= 1:
        raise ValueError(
            'Dropout probability should be valued BETWEEN 0 (included) AND 1 (not included)!')
    if args.train:
        if args.num_epochs <= 0:
            raise ValueError(
                'The maximum number of epochs in training should be POSITIVE!')
        if args.lr <= 0 or args.lr >= 1:
            raise ValueError(
                'Initial learning rate should be valued BETWEEN 0 (not included) AND 1 (not included)!')
        if args.patience <= 0 or args.patience > args.num_epochs:
            raise ValueError(
                'The maximum number of epochs to wait before early stopping should be valued BETWEEN 0 (not included) AND {:d} (included)!'.format(args.num_epochs))
        if args.lod < 0:
            raise ValueError(
                'The limit of variation in accuracy increase should be NONNEGATIVE!')
    if (args.task == 'mul' and args.dataset == 'v1d15C1' or args.task != 'mul') and args.tokenization != 'ExDs':
        raise ValueError(
            'Tokenization scheme for dataset \'{:s}\' in \'{:s}\' task can only be \'ExDs\'!')
    # if args.finetune:
    #     if not (args.task == 'prem' and args.dataset in ['v1d3C1', 'v1d7C1', 'v3t3E1C1L3']) and not (args.task == 'charset' and args.dataset in ['v2t2e4F7', 'v3t3e2C1L3', 'v3t3e2F7']):
    #         raise ValueError(
    #             'Dataset \'{:s}\' in \'{:s}\' task doesn\'t support finetuning!'.format(args.dataset, args.task))

    if args.task in ['mul', 'add']:
        if args.train and args.collect + args.combine + args.accum + args.sum > 1:
            raise ValueError(
                'Only one subtask can be trained at one time in polynomial multiplication!')
        if args.accum and not args.collect and not args.combine and not args.sum:
            if args.dataset != 'C4o9':
                raise ValueError(
                    'Accumulation dataset can only be \'C4o9\'!')
            if args.dataset_size != 1000000:
                raise ValueError(
                    'The size of dataset \'C4o9\' is 1,000,000!')
        # Multiplication task
        elif args.task == 'mul':
            if args.dataset not in ['v1d1C1', 'v1d3C1', 'v1d7C1', 'v2t2E1C1', 'v3t3E1C1', 'v1d15C1', 'v2d2C1', 'v3d2C1']:
                raise ValueError(
                    'Multiplication dataset can only RANGE AMONG [\'v1d1C1\', \'v1d3C1\', \'v1d7C1\', \'v2t2E1C1\', \'v3t3E1C1\', \'v1d15C1\']!')
            if args.dataset_size != 100000:
                raise ValueError(
                    'The size of dataset \'{:s}\' is 100,000!'.format(args.dataset))
            if args.dataset == 'v1d15C1':
                if not (args.collect + args.combine + args.accum + args.sum):
                    raise ValueError(
                        'At least one subtask should be assigned in long polynomial multiplication!')
                if not args.train and (not args.collect or args.combine or not args.accum or not args.sum):
                    raise ValueError(
                        'The only legal combination of subtasks during inference in long polynomial multiplication: \'collect\' + \'accum\' + \'sum\'!')
            else:
            #     if args.dataset in ['v1d1C1', 'v2t2E1C1', 'v3t3E1C1']:
            #         if args.collect + args.combine + args.accum + args.sum:
            #             raise ValueError(
            #                 'No subtask in multiplication dataset \'{:s}\'!'.format(args.dataset))
                # else:
                #     if args.sum:
                #         raise ValueError(
                #             'Subtask \'sum\' can only be assigned in long polynomial multiplication!')
                #     if args.combine and args.accum:
                #         raise ValueError(
                #             'Subtasks \'combine\' and \'accum\' cannot be simultaneously assigned!')
                if not (args.collect + args.combine + args.accum):
                    check_model_params(args)
        else:
            if args.collect + args.combine or not args.sum:
                raise ValueError(
                    'Subtasks \'collect\' or \'combine\' cannot be assigned and subtask \'sum\' must be assigned in polynomial addition!')
            if args.dataset not in ['v1d3C2', 'v1d7C2']:
                raise ValueError(
                    'Addition dataset can only RANGE AMONG [\'v1d3C2\', \'v1d7C2\']!')
            if args.dataset_size != 100000:
                raise ValueError(
                    'The size of dataset \'{:s}\' is 100,000!'.format(args.dataset))
        if args.collect:
            check_model_params(
                args, '_collect', ' for the subtask of learning collected product')
        if args.combine:
            check_model_params(
                args, '_combine', ' for the subtask of combining collected product')
        if args.accum:
            check_model_params(
                args, '_accum', ' for integer accumulation')
        if args.sum:
            check_model_params(
                args, '_sum', ' for the subtask of learning collected sum of combined subproducts')
    else:
        if args.collect + args.combine + args.accum + args.sum:
            raise ValueError(
                'No subtask in tasks other than multiplication and addition!')
        # Pseudodivision task
        if args.task == 'prem':
            if args.prem_step not in ['0', '1', 'N']:
                raise ValueError(
                    'The number of pseudodivision steps \'seen\' by the model can only RANGE AMONG [\'0\', \'1\', \'N\']!')
            if args.dataset not in ['v1d3C1', 'v1d7C1', '_v1D3C1L3', '_v1D7C1L3', 'v2t2E1C1L3', 'v3t3E1C1L3', 'v1d7F7', '_v1D7F7', 'v2t2E1F7', 'v3t3E1F7']:
                raise ValueError(
                    'Pseudodivision dataset can only RANGE AMONG [\'v1d3C1\', \'v1d7C1\', \'_v1D3C1L3\', \'_v1D7C1L3\', \'v2t2E1C1L3\', \'v3t3E1C1L3\', \'v1d7F7\', \'_v1D7F7\', \'v2t2E1F7\', \'v3t3E1F7\']!')
            if args.dataset_size != 100000:
                raise ValueError(
                    'The size of dataset \'{:s}\' is 100,000!'.format(args.dataset))
            if args.dataset in ['v1d3C1', 'v1d7C1']:
                if args.prem_step == 'N':
                    raise ValueError(
                        'Dataset \'{:s}\' only includes one-step pseudodivision!'.format(args.dataset))
                if args.prem_step == '0' and args.detailed_spa:
                    raise ValueError(
                        'There is no scratchpad for directly learning remainder!')
            elif args.dataset in ['v1d7F7', '_v1D7F7', 'v2t2E1F7', 'v3t3E1F7']:
                if args.dataset == 'v1D7F7':
                    if not (args.prem_step == '1' and args.detailed_spa):
                        raise ValueError(
                            'Dataset \'v1D7F7\' can only configured with `prem_step` = \'1\' and `detailed_spa`!')
            elif args.prem_step in ['0', '1'] and args.detailed_spa:
                raise ValueError(
                    'Detailed scratchpad can only exist in multi-step pseudodivision in dataset \'{:s}\'!'.format(args.dataset))
        else:
            if args.dataset not in ['v2t2e2C1L3', 'v2t2e4C1L3', 'v3t3e2C1L3', 'v2t2e2F7', 'v2t2e4F7', 'v3t3e2F7', 'v3t3e4F7', 'v3t3E1F7S2']:
                raise ValueError(
                    'Characteristic set dataset can only RANGE AMONG [\'v2t2e2C1L3\', \'v2t2e4C1L3\', \'v3t3e2C1L3\', \'v2t2e2F7\', \'v2t2e4F7\', \'v3t3e2F7\', \'v3t3e4F7\']!')
            # if args.dataset_size != 100000:
            #     raise ValueError(
            #         'The size of dataset \'{:s}\' is 100,000!'.format(args.dataset))
        check_model_params(args)
    return args


class Arguments:
    r'''Base class for data processing, model training and inference.'''

    def __init__(self, args: Namespace) -> None:
        r'''Assigns arguments.`'''
        '''
            Extract from keyboard input
        '''
        self.task = args.task
        self.dataset = args.dataset
        self.dataset_size = args.dataset_size
        self.train = args.train
        self.valid = args.valid
        self.batch_size = args.batch_size
        self.tokenization = args.tokenization
        self.model = args.model
        self.drop_prob = args.drop_prob
        self.log = args.log
        self.unsigned = getattr(args, 'unsigned', False)
        self.accum_checkpoint = getattr(args, 'accum_checkpoint', '')
        self.collect_checkpoint = getattr(args, 'collect_checkpoint', '')
        self.finetune = args.finetune
        self.no_freeze = args.no_freeze
        if args.train:
            self.num_epochs = args.num_epochs
            self.lr = args.lr
            self.patience = args.patience
            self.lod = args.lod
        if args.task == 'mul' and not (args.collect + args.combine + args.accum + args.sum) or args.task in ['prem', 'charset']:
            self.pe_scheme = args.pe_scheme
            self.emb_dim = args.emb_dim
            self.num_heads = args.num_heads
            self.num_ffn_layers = args.num_ffn_layers
            self.hidden_dim = args.hidden_dim
            self.activate_fn = args.activate_fn
            self.norm_scheme = args.norm_scheme
            self.num_encoder_layers = args.num_encoder_layers
            self.num_decoder_layers = args.num_decoder_layers
        if args.task in ['mul', 'add']:
            self.accum = args.accum
            if args.accum:
                self.pe_scheme_accum = args.pe_scheme_accum
                self.emb_dim_accum = args.emb_dim_accum
                self.num_heads_accum = args.num_heads_accum
                self.num_ffn_layers_accum = args.num_ffn_layers_accum
                self.hidden_dim_accum = args.hidden_dim_accum
                self.activate_fn_accum = args.activate_fn_accum
                self.norm_scheme_accum = args.norm_scheme_accum
                self.num_decoder_layers_accum = args.num_decoder_layers_accum
            self.sum = args.sum
            if args.sum:
                self.pe_scheme_sum = args.pe_scheme_sum
                self.emb_dim_sum = args.emb_dim_sum
                self.num_heads_sum = args.num_heads_sum
                self.num_ffn_layers_sum = args.num_ffn_layers_sum
                self.hidden_dim_sum = args.hidden_dim_sum
                self.activate_fn_sum = args.activate_fn_sum
                self.norm_scheme_sum = args.norm_scheme_sum
                self.num_encoder_layers_sum = args.num_encoder_layers_sum
                self.num_decoder_layers_sum = args.num_decoder_layers_sum
            if args.task == 'mul':
                self.collect = args.collect
                if args.collect:
                    self.pe_scheme_collect = args.pe_scheme_collect
                    self.emb_dim_collect = args.emb_dim_collect
                    self.num_heads_collect = args.num_heads_collect
                    self.num_ffn_layers_collect = args.num_ffn_layers_collect
                    self.hidden_dim_collect = args.hidden_dim_collect
                    self.activate_fn_collect = args.activate_fn_collect
                    self.norm_scheme_collect = args.norm_scheme_collect
                    self.num_encoder_layers_collect = args.num_encoder_layers_collect
                    self.num_decoder_layers_collect = args.num_decoder_layers_collect
                self.combine = args.combine
                if args.combine:
                    self.pe_scheme_combine = args.pe_scheme_combine
                    self.emb_dim_combine = args.emb_dim_combine
                    self.num_heads_combine = args.num_heads_combine
                    self.num_ffn_layers_combine = args.num_ffn_layers_combine
                    self.hidden_dim_combine = args.hidden_dim_combine
                    self.activate_fn_combine = args.activate_fn_combine
                    self.norm_scheme_combine = args.norm_scheme_combine
                    self.num_encoder_layers_combine = args.num_encoder_layers_combine
                    self.num_decoder_layers_combine = args.num_decoder_layers_combine
        elif args.task == 'prem':
            self.prem_step = args.prem_step
            self.detailed_spa = args.detailed_spa
        '''
            Vocabulary
        '''
        # Special, operator tokens
        spec = ['<pad>', '<s>', '</s>']
        ops = ['+', '-', '*', '^']
        # Number tokens
        if args.tokenization[2:] == 'Ds':
            nums = ['{:d}'.format(i) for i in range(10)]
        else:
            nums = ['{:d}'.format(i) for i in range(
                1, 649 if args.dataset == 'v1d7C1' else 325 if args.dataset in ['v1d3C1', 'v3t3E1C1'] else 163)]
        # Variable tokens
        var = ['x'] if args.dataset[:2] == 'v1' or args.dataset[:3] == '_v1' else [
            'x{:d}'.format(i) for i in range(eval(args.dataset[1]))]
        # Vocabularies
        if args.task == 'mul':
            if args.dataset == 'v1d15C1' or args.task == 'add':
                self.vocab = spec + ['<op>', '(', ')'] + ops + var + nums
                self.vocab_collect = self.vocab
                self.vocab_accum = spec + ['>', '+', '-'] + nums
            else:
                if args.collect + args.combine + args.accum != 1:
                    self.vocab = spec + ['<op>'] + ops + var + nums
                if args.collect + args.combine:
                    self.vocab_collect = spec + \
                        ['<op>', '(', ')'] + ops + var + nums
                if args.accum:
                    self.vocab_accum = spec + ['>', '+', '-'] + nums
        elif self.task == 'add':
            self.vocab = spec + ['<op>', '(', ')'] + ops + var + nums
            if args.accum:
                self.vocab_accum = spec + ['>', '+', '-'] + nums
        else:
            spec += ['<op>', '>']
            self.vocab = spec + ops + var + nums
        if args.model == 'polynormer':
            spec = ['<pad>', '<s>', '</s>']
            nums = ['+', '-'] + ['c{:d}'.format(c) for c in range(10)]
            nums.extend(['e{:d}'.format(e) for e in range(10)])
            var = ['x']
            self.vocab = spec + nums + var
        '''
            Maximum number of cache slots for different datasets to save GPU memory
        '''
        if args.task in ['mul', 'add']:
            if args.accum:
                self.max_seq_len_accum = 111
            if args.dataset == 'v1d3C2':
                self.max_in_seq_len_sum = 51
                self.max_out_seq_len_sum = 46
            elif args.dataset == 'v1d7C2':
                self.max_in_seq_len_sum = 107
                self.max_out_seq_len_sum = 94
            if args.dataset == 'v1d1C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 19
                    self.max_out_seq_len = 20
                if args.collect:
                    self.max_in_seq_len_collect = 19
                    self.max_out_seq_len_collect = 24
                if args.combine:
                    self.max_in_seq_len_combine = 24
                    self.max_out_seq_len_combine = 20
            elif args.dataset == 'v1d3C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 43
                    self.max_out_seq_len = 52
                if args.collect:
                    self.max_in_seq_len_collect = 43
                    self.max_out_seq_len_collect = 84
                if args.combine:
                    self.max_in_seq_len_combine = 84
                    self.max_out_seq_len_combine = 52
            elif args.dataset == 'v1d7C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 91
                    self.max_out_seq_len = 121
                if args.collect:
                    self.max_in_seq_len_collect = 91
                    self.max_out_seq_len_collect = 281
                if args.combine:
                    self.max_in_seq_len_combine = 281
                    self.max_out_seq_len_combine = 121
            elif args.dataset == 'v2t2E1C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 43
                    self.max_out_seq_len = 54
                if args.collect:
                    self.max_in_seq_len_collect = 43
                    self.max_out_seq_len_collect = 54
                if args.combine:
                    self.max_in_seq_len_combine = 54
                    self.max_out_seq_len_combine = 54
            elif args.dataset == 'v3t3E1C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 87
                    self.max_out_seq_len = 164
                if args.collect:
                    self.max_in_seq_len_collect = 87
                    self.max_out_seq_len_collect = 164
                if args.combine:
                    self.max_in_seq_len_combine = 164
                    self.max_out_seq_len_combine = 164
            elif args.dataset == 'v3d2C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 139
                    self.max_out_seq_len = 378
                if args.collect:
                    self.max_in_seq_len_collect = 139
                    self.max_out_seq_len_collect = 604
                if args.combine:
                    self.max_in_seq_len_combine = 604
                    self.max_out_seq_len_combine = 378
            elif args.dataset == 'v2d2C1':
                if not args.collect and not args.combine:
                    self.max_in_seq_len = 75
                    self.max_out_seq_len = 139
                if args.collect:
                    self.max_in_seq_len_collect = 75
                    self.max_out_seq_len_collect = 214
                if args.combine:
                    self.max_in_seq_len_combine = 214
                    self.max_out_seq_len_combine = 139
            else:
                if args.collect:
                    self.max_in_seq_len_collect = 87
                    self.max_out_seq_len_collect = 164
                if args.sum:
                    self.max_in_seq_len_sum = 333
                    self.max_out_seq_len_sum = 350
        elif args.task == 'prem':
            if args.dataset == 'v1d3C1':
                self.max_in_seq_len = 51
                if args.prem_step == '0':
                    self.max_out_seq_len = 47
                else:
                    self.max_out_seq_len = 66
            elif args.dataset == 'v1d7C1':
                self.max_in_seq_len = 107
                if args.prem_step == '0':
                    self.max_out_seq_len = 107
                elif args.detailed_spa:
                    self.max_out_seq_len = 768
                else:
                    self.max_out_seq_len = 130
            elif args.dataset == '_v1D3C1L3':
                self.max_in_seq_len = 39
                self.max_out_seq_len = 768
            elif args.dataset == '_v1D7C1L3':
                self.max_in_seq_len = 87
                self.max_out_seq_len = 768
            elif args.dataset in ['v2t2E1C1L3', 'v2t2E1F7']:
                self.max_in_seq_len = 43
                if args.detailed_spa:
                    self.max_out_seq_len = 1024
                else:
                    self.max_out_seq_len = 768
            elif args.dataset in ['v3t3E1C1L3', 'v3t3E1F7']:
                self.max_in_seq_len = 87
                self.max_out_seq_len = 768
            elif args.dataset == 'v1d7F7':
                self.max_in_seq_len = 91
                self.max_out_seq_len = 768
            elif args.dataset == '_v1D7F7':
                self.max_in_seq_len = 87
                self.max_out_seq_len = 768
        else:
            if args.dataset[:2] == 'v2':
                self.max_in_seq_len = 43
                self.max_out_seq_len = 1024
            else:
                self.max_in_seq_len = 87
                if args.dataset == 'v3t3e2F7':
                    self.max_out_seq_len = 1280
                elif args.dataset == 'v3t3E1F7S2':
                    self.max_out_seq_len = 1920
                else:
                    self.max_out_seq_len = 768
        '''
            To support unsafe data loading in PyTorch with 2.6 or higher version
        '''
        torch_version = eval(match(r'(\d+\.\d+)', __version__).group(1))
        self.new_torch = torch_version >= 2.6
        '''
            Finetune model paths
        '''
        if args.finetune:
            dir = join(abspath(dirname(dirname(__file__))), 'model')
            model_name = 'PE_' + args.pe_scheme + '_D_' + str(args.emb_dim) + '_H_' + str(args.num_heads) + '_NE_' + str(args.num_encoder_layers) + '_DE_' + str(args.num_decoder_layers) + '.pth'
            if self.dataset == 'v1d3C1':
                self.ft_model_path = join(dir, 'mul', 'v1d3C1_100000', 'ExDs', 'collect', model_name)
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '(', ')', '+', '-', '*', '^', 'x'] + ['{:d}'.format(i) for i in range(10)]
            elif self.dataset == 'v1d7C1':
                self.ft_model_path = join(dir, 'mul', 'v1d7C1_100000', 'ExDs', 'collect', model_name)
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '(', ')', '+', '-', '*', '^', 'x'] + ['{:d}'.format(i) for i in range(10)]
            elif self.dataset == 'v2t2E1C1L3':
                self.ft_model_path = join(dir, 'mul', 'v2t2E1C1_100000', 'ExDs', model_name)
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '+', '-', '*', '^'] + ['x0', 'x1'] + ['{:d}'.format(i) for i in range(10)]
            elif self.dataset == 'v3t3E1C1L3':
                self.ft_model_path = join(dir, 'mul', 'v3t3E1C1_100000', 'ExDs', model_name)
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '+', '-', '*', '^'] + ['x0', 'x1', 'x2'] + ['{:d}'.format(i) for i in range(10)]
            elif self.dataset == 'v2t2E1F7':
                self.ft_model_path = join(dir, 'prem', 'v2t2E1C1L3_100000', 'detailed_spa', model_name)
            elif self.dataset == 'v2t2e4F7' and self.task == 'charset':
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '+', '-', '*', '^'] + ['x0', 'x1', 'x2'] + ['{:d}'.format(i) for i in range(10)] + ['>']
                self.ft_model_path = join(dir, 'prem', 'v2t2E1F7_100000', 'ft2', model_name)
            elif self.dataset == 'v3t3E1F7':
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '+', '-', '*', '^'] + ['x0', 'x1', 'x2'] + ['{:d}'.format(i) for i in range(10)]
                self.ft_model_path = join(dir, 'prem', 'v3t3E1C1L3_100000', 'ft2', model_name)
            else:
                self.vocab = ['<pad>', '<s>', '</s>', '<op>', '+', '-', '*', '^'] + ['x0', 'x1', 'x2'] + ['{:d}'.format(i) for i in range(10)] + ['>']
                if self.dataset == 'v3t3e2C1L3':
                    self.ft_model_path = join(dir, 'prem', 'v3t3E1C1L3_100000', 'ft2', model_name)
                elif self.dataset == 'v3t3e2F7':
                    self.ft_model_path = join(dir, 'prem', 'v3t3E1F7_100000', 'ft2', model_name)
                elif self.dataset == 'v3t3E1F7S2':
                    self.ft_model_path = join(dir, 'prem', 'v3t3E1F7_100000', 'ft2', model_name)

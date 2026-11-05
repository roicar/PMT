'''
    Running file\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 18 Feb 2026
'''
from warnings import filterwarnings
from data_utils import assign_arguments
from os.path import join, dirname
from utils import generate_folder
import sys
from model_utils import Runner

if __name__ == '__main__':
    # Ignore all warnings
    filterwarnings('ignore')
    args = assign_arguments()
    if args.log:
        base_dir = join(dirname(__file__), 'log')
        if args.task == 'mul' and args.dataset != 'v1d15C1':
            data_spec = join(args.dataset + '_' +
                             str(args.dataset_size), args.tokenization)
        else:
            data_spec = args.dataset + '_' + str(args.dataset_size)
        suffix = '_train.log' if args.train else '_valid.log' if args.valid else '_test.log'
        if getattr(args, 'collect', False) + getattr(args, 'combine', False) + getattr(args, 'accum', False) + getattr(args, 'sum', False):
            if getattr(args, 'collect', False):
                collect_model_prefix = 'PE_' + args.pe_scheme_collect + '_D_' + str(args.emb_dim_collect) + '_H_' + str(
                    args.num_heads_collect) + '_NE_' + str(args.num_encoder_layers_collect) + '_DE_' + str(args.num_decoder_layers_collect)
            if getattr(args, 'combine', False):
                combine_model_prefix = 'PE_' + args.pe_scheme_combine + '_D_' + str(args.emb_dim_combine) + '_H_' + str(
                    args.num_heads_combine) + '_NE_' + str(args.num_encoder_layers_combine) + '_DE_' + str(args.num_decoder_layers_combine)
            if getattr(args, 'accum', False):
                accum_model_prefix = 'PE_' + args.pe_scheme_accum + '_D_' + \
                    str(args.emb_dim_accum) + '_H_' + str(args.num_heads_accum) + \
                    '_NE_0_DE_' + str(args.num_decoder_layers_accum)
            if getattr(args, 'sum', False):
                sum_model_prefix = 'PE_' + args.pe_scheme_sum + '_D_' + str(args.emb_dim_sum) + '_H_' + str(
                    args.num_heads_sum) + '_NE_' + str(args.num_encoder_layers_sum) + '_DE_' + str(args.num_decoder_layers_sum)
        else:
            model_prefix = 'PE_' + args.pe_scheme + '_D_' + \
                str(args.emb_dim) + '_H_' + str(args.num_heads) + \
                '_NE_' + str(args.num_encoder_layers) + '_DE_' + str(args.num_decoder_layers)
        if getattr(args, 'collect', False) + getattr(args, 'combine', False) + getattr(args, 'accum', False) + getattr(args, 'sum', False) == 1:
            if getattr(args, 'collect', False):
                log_folder = join(base_dir, 'mul', data_spec, 'collect')
                log_path = join(log_folder, collect_model_prefix + suffix)
            if getattr(args, 'combine', False):
                log_folder = join(base_dir, 'mul', data_spec, 'combine')
                log_path = join(log_folder, combine_model_prefix + suffix)
            if getattr(args, 'accum', False):
                log_folder = join(base_dir, 'accum', args.dataset +
                                  '_' + str(args.dataset_size))
                log_path = join(log_folder, accum_model_prefix + suffix)
            if getattr(args, 'sum', False):
                log_folder = join(base_dir, 'mul', data_spec, 'sum')
                log_path = join(log_folder, sum_model_prefix + suffix)
        elif getattr(args, 'collect', False):
            if args.combine:
                log_folder = join(base_dir, 'mul', data_spec,
                                  'collect_and_combine')
                log_path = join(log_folder, 'Collect_' + collect_model_prefix +
                                '_Combine_' + combine_model_prefix + suffix)
            elif args.sum:
                log_folder = join(base_dir, 'mul', data_spec,
                                  'collect_and_accum_and_sum')
                log_path = join(log_folder, 'Collect_' + collect_model_prefix +
                                '_Accum_' + accum_model_prefix + '_Sum_' + sum_model_prefix + suffix)
            else:
                log_folder = join(base_dir, 'mul', data_spec,
                                  'collect_and_accum')
                log_path = join(log_folder, 'Collect_' + collect_model_prefix +
                                '_Accum_' + accum_model_prefix + suffix)
        elif getattr(args, 'accum', False) and getattr(args, 'sum', False):
            log_folder = join(base_dir, 'mul', data_spec, 'sum_and_accum')
            log_path = join(log_folder, 'Sum_' + sum_model_prefix +
                            '_Accum_' + accum_model_prefix + suffix)
        else:
            log_folder = join(base_dir, args.task, data_spec)
            if args.finetune:
                if args.no_freeze:
                    log_folder = join(log_folder, 'ft2')
                else:
                    log_folder = join(log_folder, 'ft')
            elif args.task == 'prem' and args.dataset not in ['v1d7F7', '_v1D7F7', 'v2t2E1F7', 'v3t3E1F7']:
                step_folder = args.prem_step + '-step' if not args.detailed_spa else 'detailed_spa'
                log_folder = join(log_folder, step_folder)
            log_path = join(log_folder, model_prefix + suffix)
        generate_folder(log_folder)

        def save_as_log():
            r'''Save printed messages as a logging file.'''

            class Logger(object):
                r'''A logger object for calling `stdout` to write a file.'''

                def __init__(self, path: str = log_path) -> None:
                    self.terminal = sys.stdout
                    self.log = open(path, 'w', encoding='utf-8')

                def write(self, message: str) -> None:
                    r'''Write standard output message to the logging file.

                    Args:
                        message (str) : A piece of standard output message (when calling `print()` in Python).
                    '''
                    self.terminal.write(message)
                    self.log.write(message)

                def flush(self):
                    r'''Necessary for object declaration, but useless here.'''
                    pass

            # Replace `stdout` by a `Logger` object
            sys.stdout = Logger()

        save_as_log()

    runner = Runner(args=args)
    runner.run(args=args)

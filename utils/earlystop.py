'''
    Early stopping strategy when training a Transformer-based model\\
    Coded and annoated by Yuxuan SONG
    Last modified time: 22 Dec 2025
'''
from torch import save


class EarlyStopping:
    '''Early stops the training if validation accuracy doesn't improve after a given patience.'''

    def __init__(self, patience: int, lod: float, path: str, max_seq_acc: float = 0, max_token_acc: float = 0, verbose: bool = True):
        '''
        Args:
            patience (int): The maximum number of epochs to wait before early stopping.
            lod (float): Limit of variation in accuracy increase.
            path (str): Path of storing the best trained model.
            max_seq_acc (float): Initial maximum sequence-level teacher-forcing accuracy on validation dataset (default: 0).
            max_token_acc (float): Initial maximum token-level teacher-forcing accuracy on validation dataset (default: 0).
            verbose (bool): Whether verbose internal processes (default: True).
        '''
        self.patience = patience
        self.lod = lod
        self.path = path
        self.max_seq_acc = max_seq_acc
        self.max_token_acc = max_token_acc
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False

    def __call__(self, seq_acc: float, token_acc: float, model) -> None:
        r'''Implementation of early stopping strategy.

        Args:
            seq_acc (float): Sequence-level accuracy on validation dataset.
            token_acc (float): Token-level accuracy on validation dataset.
            model: Model at current epoch.
        '''
        variation = seq_acc - self.max_seq_acc if seq_acc else token_acc - self.max_token_acc
        if variation > self.lod:
            print('{:s} Acc. increased from {:.4f} to {:.4f}'.format('Seq' if seq_acc else 'Token',
                  self.max_seq_acc if seq_acc else self.max_token_acc, seq_acc if seq_acc else token_acc))
            save(model.state_dict(), self.path)
            if self.verbose:
                print('The best model is saved in \'{:s}\'.'.format(self.path))
            if seq_acc:
                self.max_seq_acc = seq_acc
            else:
                self.max_token_acc = token_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {:d} out of {:d}'.format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

'''
    Utilities for data loaders\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 28 Jan 2026
'''
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Union
from torch import Tensor, cuda, load, save, flip, cat
from data_utils.storage import SequenceData, PosCpData, LongSequenceMulData
from torch.nn.utils.rnn import pad_sequence
from unsigned.data_utils.token import Tokenization
from os.path import join, exists
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData, Batch


class S2SDataset(Dataset):
    r'''Dataset of sequence to sequence pairs for encoder-decoder architecture.'''

    def __init__(self, data: SequenceData) -> None:
        r'''Initialization of the class.'''
        self.data = data

    def __len__(self) -> None:
        r'''Obtain the length of data pairs.'''
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        r'''Get items in a data pair'''
        return self.data[idx].in_vocabid, self.data[idx].out_vocabid


def S2S_collate_fn(batch: List) -> Tuple[Tensor]:
    r'''Collation function for batch sequence to sequence data pairs.'''
    in_seqs, out_seqs = zip(*batch)
    in_padded = pad_sequence(in_seqs, batch_first=True)
    in_mask = in_padded == 0
    out_padded = pad_sequence(out_seqs, batch_first=True)
    out_mask = out_padded == 0
    return in_padded, in_mask, out_padded, out_mask


class SDataset(Dataset):
    r'''Dataset of sequences for decoder-only architecture.'''

    def __init__(self, data: SequenceData) -> None:
        self.data = data

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx].in_vocabid + self.data[idx].out_vocabid


def S_collate_fn(batch: List) -> Tuple[Tensor]:
    seqs = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True)
    mask = padded == 0
    return padded, mask


class SeparateSDataset(Dataset):
    r'''Dataset of sequences for decoder-only architecture with separately padded prompts and responses.'''

    def __init__(self, data: SequenceData) -> None:
        self.data = data

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.data[idx].in_vocabid, self.data[idx].out_vocabid


def SS_collate_fn(batch: List) -> Tuple[Tensor]:
    prompts, responses = zip(*batch)
    prompts = [flip(prompt, dims=[0]) for prompt in prompts]
    prompts = pad_sequence(prompts, batch_first=True)
    prompts = flip(prompts, dims=[1])
    responses = pad_sequence(responses, batch_first=True)
    padded = cat([prompts, responses], dim=1)
    mask = padded == 0
    return padded, mask


class PosCpDataset(Dataset):
    r'''Dataset of integer accumulation sequences with multi-level position coupling and scratchpad.'''

    def __init__(self, data: PosCpData) -> None:
        self.data = data

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.data[idx].operand, self.data[idx].operand_id1, self.data[idx].operand_id2, self.data[idx].result, self.data[idx].result_id1, self.data[idx].result_id2, self.data[idx].operand_rmov


def PosCp_collate_fn(batch: List) -> Tuple[Tensor]:
    operand, operand_id1, operand_id2, result, result_id1, result_id2, operand_rmov = zip(
        *batch)
    operand = pad_sequence(operand, batch_first=True)
    operand_mask = operand == 0
    operand_id1 = pad_sequence(operand_id1, batch_first=True)
    operand_id2 = pad_sequence(operand_id2, batch_first=True)
    result = pad_sequence(result, batch_first=True)
    result_mask = result[:, :-1] == 0
    result_id1 = pad_sequence(result_id1, batch_first=True)
    result_id2 = pad_sequence(result_id2, batch_first=True)
    operand_rmov = pad_sequence(operand_rmov, batch_first=True)
    return operand, operand_mask, operand_id1, operand_id2, result, result_mask, result_id1, result_id2, operand_rmov


class LongMulDataset(Dataset):
    r'''Dataset of long sequence polynomial multiplication samples.'''

    def __init__(self, data: LongSequenceMulData) -> None:
        self.data = data

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.data[idx].f0g0, self.data[idx].f0g1, self.data[idx].f1g0, self.data[idx].f1g1, self.data[idx].product


def LongMul_collate_fn(batch: List) -> Tuple[Tensor]:
    f0g0, f0g1, f1g0, f1g1, product = zip(*batch)
    f0g0 = pad_sequence(f0g0, batch_first=True)
    f0g0_mask = f0g0 == 0
    f0g1 = pad_sequence(f0g1, batch_first=True)
    f1g0_mask = f0g1 == 0
    f1g0 = pad_sequence(f1g0, batch_first=True)
    f1g0_mask = f1g0 == 0
    f1g1 = pad_sequence(f1g1, batch_first=True)
    f1g1_mask = f1g1 == 0
    product = pad_sequence(product, batch_first=True)
    product_mask = product == 0
    return f0g0, f0g0_mask, f0g1, f1g0_mask, f1g0, f1g0_mask, f1g1, f1g1_mask, product, product_mask
        
class PolyGraphSeqDataset:
    def __init__(self, in_graph:HeteroData, out_seq:HeteroData)->None:
        self.in_graph = in_graph
        self.out_seq = out_seq
        
    def __len__(self)->int:
        return len(self.in_graph)
    
    def __getitem__(self, idx:int)->Tuple[HeteroData, Tensor]:
        return self.in_graph[idx], self.out_seq[idx]
    
def collate_fn_graph(batch:List)->Tuple[HeteroData, Tensor]:
    in_graphs, out_seqs = zip(*batch)
    in_graphs = Batch.from_data_list(in_graphs)
    out_seqs = pad_sequence(out_seqs, True)
    out_mask = out_seqs == 0
    return in_graphs, out_seqs, out_mask


class PolyDataloader(Tokenization):
    r'''Data loaders for polynomial arithmetic.'''

    def __init__(self, **kwargs) -> None:
        r'''Directories and path of files storing data loaders.'''
        super().__init__(**kwargs)
        loader_suffix = '_loader_' + str(self.batch_size) + '.pt'
        self.train_loader_path = join(
            self.processed_data_folder, 'train' + loader_suffix)
        self.val_loader_path = join(
            self.processed_data_folder, 'val' + loader_suffix)
        self.test_loader_path = join(
            self.processed_data_folder, 'test' + loader_suffix)

    def generate_loader(self, data: List, shuffle: bool = False) -> DataLoader:
        if self.model == 'polynormer':
            return DataLoader(PolyGraphSeqDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate_fn_graph, pin_memory=cuda.is_available())
        elif not getattr(self, 'collect', False) and getattr(self, 'accum', False):
            if self.pe_scheme_accum == 'pc':
                return DataLoader(PosCpDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=PosCp_collate_fn, pin_memory=cuda.is_available())
            elif self.pe_scheme_accum in ['rpe-s', 'rpe-l']:
                return DataLoader(SeparateSDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=SS_collate_fn, pin_memory=cuda.is_available())
            return DataLoader(SDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=S_collate_fn, pin_memory=cuda.is_available())
        elif self.dataset in ['v1d15C1', 'v4t4E1C1'] and getattr(self, 'collect', False) and getattr(self, 'sum', False):
            return DataLoader(LongMulDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=LongMul_collate_fn, pin_memory=cuda.is_available())
        else:
            return DataLoader(S2SDataset(data), batch_size=self.batch_size, shuffle=shuffle, collate_fn=S2S_collate_fn, pin_memory=cuda.is_available())

    def construct(self) -> Union[Tuple[DataLoader], DataLoader, List]:
        if self.train and exists(self.train_loader_path) and exists(self.val_loader_path):
            print('Data loader for training is loaded from \'{:s}\''.format(
                self.train_loader_path))
            if self.new_torch:
                train_loader = load(self.train_loader_path, weights_only=False)
            else:
                train_loader = load(self.train_loader_path)
            print('Data loader for validation is loaded from \'{:s}\''.format(
                self.val_loader_path))
            if self.new_torch:
                val_loader = load(self.val_loader_path, weights_only=False)
            else:
                val_loader = load(self.val_loader_path)
        elif not self.train and self.valid and exists(self.val_loader_path):
            print('Data loader for validation is loaded from \'{:s}\''.format(
                self.val_loader_path))
            if self.new_torch:
                val_loader = load(self.val_loader_path, weights_only=False)
            else:
                val_loader = load(self.val_loader_path)
        elif not self.train and not self.valid and exists(self.test_loader_path):
            print('Data loader for testing is loaded from \'{:s}\''.format(
                self.test_loader_path))
            if self.new_torch:
                test_loader = load(self.test_loader_path, weights_only=False)
            else:
                test_loader = load(self.test_loader_path)
        else:
            seed_everything(0)
            data = self.tokenize()
            if self.train:
                train_data = data[:int(self.dataset_size * 0.7)]
                train_loader = self.generate_loader(train_data)
                save(train_loader, self.train_loader_path)
                print('Data loader for training is saved in \'{:s}\'.'.format(
                    self.train_loader_path))
                val_data = data[int(self.dataset_size * 0.7)  # noqa
                                    :int(self.dataset_size * 0.8)]  # noqa
                val_loader = self.generate_loader(val_data)
                save(val_loader, self.val_loader_path)
                print('Data loader for validation is saved in \'{:s}\'.'.format(
                    self.val_loader_path))
            elif self.valid:
                val_data = data[int(self.dataset_size * 0.7)  # noqa
                                    :int(self.dataset_size * 0.8)]  # noqa
                val_loader = self.generate_loader(val_data)
                save(val_loader, self.val_loader_path)
                print('Data loader for validation is saved in \'{:s}\'.'.format(
                    self.val_loader_path))
            else:
                test_data = data[-int(self.dataset_size * 0.2):]
                test_loader = self.generate_loader(test_data)
                save(test_loader, self.test_loader_path)
                print('Data loader for testing is saved in \'{:s}\'.'.format(
                    self.test_loader_path))
        if self.train:
            return train_loader, val_loader
        elif self.valid:
            return val_loader
        else:
            return test_loader

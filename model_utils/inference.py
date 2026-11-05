'''
    Program runner\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 18 Feb 2026
'''
from typing import Final, Union, List
from data_utils import Arguments, PolyDataloader
from os.path import join, dirname, exists
from utils import generate_folder, EarlyStopping
from data_utils import Int32X, ids_to_str, fetch_res_in_scp, InferenceResults, separate_collected_coeffs, accum_list_dataloader
from model_utils.transformer import Transformer
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from torch_optimizer import Ranger
from torch.nn import CrossEntropyLoss
from torch import cat, no_grad, save, cuda, load, tensor, ones, zeros, where, unique, arange, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from model_utils.polynormer import Polynormer

NUM_DIGITS: Final = 32
NUM_OPERANDS: Final = 32


class Runner(Arguments):
    r'''Class for training and inference of Transformer-based models.'''

    def __init__(self, **kwargs) -> None:
        r'''Initialization of the class.'''
        super().__init__(**kwargs)
        # Folder, file path of model saving and loading
        base_dir = join(dirname(dirname(__file__)), 'model')
        data_spec = self.dataset + '_' + str(self.dataset_size)
        if self.task == 'mul' and self.dataset != 'v1d15C1':
            data_spec = join(data_spec, self.tokenization)
        if self.task in ['mul', 'add']:
            if getattr(self, 'collect', False) + getattr(self, 'combine', False) + self.accum + self.sum:
                if getattr(self, 'collect', False):
                    collect_folder = join(
                        base_dir, 'mul', data_spec, 'collect')
                    generate_folder(collect_folder)
                    self.collect_model_path = join(collect_folder, 'PE_' + self.pe_scheme_collect + '_D_' + str(self.emb_dim_collect) + '_H_' + str(
                        self.num_heads_collect) + '_NE_' + str(self.num_encoder_layers_collect) + '_DE_' + str(self.num_decoder_layers_collect) + '.pth')
                if getattr(self, 'combine', False):
                    combine_folder = join(
                        base_dir, 'mul', data_spec, 'combine')
                    generate_folder(combine_folder)
                    self.combine_model_path = join(combine_folder, 'PE_' + self.pe_scheme_combine + '_D_' + str(self.emb_dim_combine) + '_H_' + str(
                        self.num_heads_combine) + '_NE_' + str(self.num_encoder_layers_combine) + '_DE_' + str(self.num_decoder_layers_combine) + '.pth')
                if self.accum:
                    accum_folder = join(base_dir, 'accum', 'C4o9_1000000')
                    generate_folder(accum_folder)
                    self.accum_model_path = join(accum_folder, 'PE_' + self.pe_scheme_accum + '_D_' + str(
                        self.emb_dim_accum) + '_H_' + str(self.num_heads_accum) + '_NE_0_DE_' + str(self.num_decoder_layers_accum) + '.pth')
                if self.sum:
                    sum_folder = join(base_dir, 'mul', data_spec, 'sum')
                    generate_folder(sum_folder)
                    self.sum_model_path = join(sum_folder, 'PE_' + self.pe_scheme_sum + '_D_' + str(
                        self.emb_dim_sum) + '_H_' + str(self.num_heads_sum) + '_NE_0_DE_' + str(self.num_decoder_layers_sum) + '.pth')
            else:
                model_folder = join(dirname(dirname(__file__)),
                                    'model', self.task, data_spec)
                generate_folder(model_folder)
                self.model_path = join(model_folder, 'PE_' + self.pe_scheme + '_D_' + str(self.emb_dim) + '_H_' + str(
                self.num_heads) + '_NE_' + str(self.num_encoder_layers) + '_DE_' + str(self.num_decoder_layers) + '.pth')

        else:
            model_spec = 'PE_' + self.pe_scheme + '_D_' + str(self.emb_dim) + '_H_' + str(
                self.num_heads) + '_NE_' + str(self.num_encoder_layers) + '_DE_' + str(self.num_decoder_layers) + '.pth'
            if self.finetune:
                if self.no_freeze:
                    model_folder = join(dirname(dirname(__file__)),
                                    'model', self.task, data_spec, 'ft2')
                else:
                    model_folder = join(dirname(dirname(__file__)),
                                    'model', self.task, data_spec, 'ft')
            elif self.task == 'prem' and self.dataset not in ['v1d7F7', '_v1D7F7', 'v2t2E1F7', 'v3t3E1F7']:
                step_folder = self.prem_step + '-step' if not self.detailed_spa else 'detailed_spa'
                model_folder = join(dirname(dirname(__file__)),
                                    'model', self.task, data_spec, step_folder)
            else:
                model_folder = join(dirname(dirname(__file__)),
                                    'model', self.task, data_spec)
            generate_folder(model_folder)
            self.model_path = join(model_folder, model_spec)

    def training(self, model: Transformer, train_loader: DataLoader, val_loader: DataLoader, model_path: str) -> None:
        r'''Training the model.'''
        seed_everything(0)
        device = next(model.parameters()).device
        val_size = int(self.dataset_size * 0.1)

        optimizer = Ranger(model.parameters(), lr=self.lr)
        loss_fn = CrossEntropyLoss(ignore_index=0)  # Ignore paddings
        stop_dectector = EarlyStopping(self.patience, self.lod, model_path)

        for epoch in range(self.num_epochs):
            train_loss = 0
            correct_seqs = 0
            token_acc = 0

            # Integer accumulation subtask with position coupling and scratchpad
            if getattr(self, 'pe_scheme_accum', '') == 'pc':
                # Teacher-forcing training
                model.train()
                for batch, (prompt, prompt_mask, prompt_id1, prompt_id2, response, response_mask, response_id1, response_id2, prompt_rmov) in enumerate(train_loader):
                    sequence = cat([prompt, response[:, :-1]],
                                   dim=-1).to(device)
                    posid1 = cat([prompt_id1, response_id1], dim=-1).to(device)
                    posid2 = cat([prompt_id2, response_id2], dim=-1).to(device)
                    mask = cat([prompt_mask, response_mask], dim=-1).to(device)
                    label = cat([prompt_rmov, response[:, 1:]],
                                dim=-1).to(device)
                    logits = model(x=sequence, x_mask=mask, posid1=posid1,
                                   posid2=posid2, teacher_forcing=True)
                    loss = loss_fn(logits.flatten(0, 1), label.flatten())
                    # Zeroize gradients
                    optimizer.zero_grad()
                    # Loss backpropagation
                    loss.backward()
                    # Gradient clipping
                    clip_grad_norm_(model.parameters(), max_norm=5)
                    # Update parameters
                    optimizer.step()
                    # Accumulate loss
                    train_loss += loss.item()
                # Evaluate on validation data loader with teacher-forcing
                model.eval()
                for _, (prompt, prompt_mask, prompt_id1, prompt_id2, response, response_mask, response_id1, response_id2, prompt_rmov) in enumerate(val_loader):
                    np = prompt.size(1)
                    sequence = cat([prompt, response[:, :-1]],
                                   dim=-1).to(device)
                    posid1 = cat([prompt_id1, response_id1], dim=-1).to(device)
                    posid2 = cat([prompt_id2, response_id2], dim=-1).to(device)
                    mask = cat([prompt_mask, response_mask], dim=-1).to(device)
                    label = response[:, 1:].to(device)
                    with no_grad():
                        logits = model(
                            x=sequence, x_mask=mask, posid1=posid1, posid2=posid2, teacher_forcing=True)
                    predict = logits[:, np:].argmax(dim=-1)
                    # The number of valid (non-padding) tokens in each response
                    num_tokens = (label != 0).sum(dim=1)
                    # The number of tokens correctly predicted
                    correct_tokens = ((predict == label) &
                                      (label != 0)).sum(dim=1)
                    # Accumulate accuracy of prediction on tokens in each sequence
                    token_acc += (correct_tokens / num_tokens).sum()
                    # Accumulate the number of sequences in which all tokens are correctly predicted
                    correct_seqs += (correct_tokens == num_tokens).sum()

            # Integer accumulation subtask with other position embedding schemes
            elif model.decoder_only:
                model.train()
                for batch, (vocabid, mask) in enumerate(train_loader):
                    vocabid = vocabid.to(device)
                    mask = mask[:, :-1].to(device)
                    logits = model(x=vocabid[:, :-1],
                                   x_mask=mask, teacher_forcing=True)
                    loss = loss_fn(logits.flatten(0, 1),
                                   vocabid[:, 1:].flatten())
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=5)
                    optimizer.step()
                    train_loss += loss.item()
                model.eval()
                for vocabid, mask in val_loader:
                    vocabid = vocabid.to(next(model.parameters()).device)
                    mask = mask[:, :-1].to(next(model.parameters()).device)
                    with no_grad():
                        logits = model(
                            x=vocabid[:, :-1], x_mask=mask, teacher_forcing=True)
                    predicts = logits.argmax(dim=-1)
                    targets = (out_vocabid * out_vocabid.sgn())[:, 1:]
                    num_tokens = (targets != 0).sum(dim=1)
                    correct_tokens = ((predicts == targets) &
                                      (targets != 0)).sum(dim=1)
                    token_acc += (correct_tokens / num_tokens).sum()
                    correct_seqs += (correct_tokens == num_tokens).sum()

            # Normal APE models
            else:
                model.train()
                for batch, (in_vocabid, in_mask, out_vocabid, out_mask) in enumerate(train_loader):
                    in_vocabid = in_vocabid.to(next(model.parameters()).device)
                    in_mask = in_mask.to(next(model.parameters()).device)
                    out_vocabid = out_vocabid.to(
                        next(model.parameters()).device)
                    out_mask = out_mask[:, :-
                                        1].to(next(model.parameters()).device)
                    logits = model(
                        out_vocabid[:, :-1], out_mask, in_vocabid, in_mask, teacher_forcing=True)
                    loss = loss_fn(logits.flatten(0, 1),
                                   out_vocabid[:, 1:].flatten())
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=5)
                    optimizer.step()
                    train_loss += loss.item()
                model.eval()
                for in_vocabid, in_mask, out_vocabid, out_mask in val_loader:
                    in_vocabid = in_vocabid.to(next(model.parameters()).device)
                    in_mask = in_mask.to(next(model.parameters()).device)
                    out_vocabid = out_vocabid.to(
                        next(model.parameters()).device)
                    out_mask = out_mask[:, :-
                                        1].to(next(model.parameters()).device)
                    with no_grad():
                        logits = model(
                            out_vocabid[:, :-1], out_mask, in_vocabid, in_mask, teacher_forcing=True)
                    predicts = logits.argmax(dim=-1)
                    targets = (out_vocabid * out_vocabid.sgn())[:, 1:]
                    num_tokens = (targets != 0).sum(dim=1)
                    correct_tokens = ((predicts == targets) &
                                      (targets != 0)).sum(dim=1)
                    token_acc += (correct_tokens / num_tokens).sum()
                    correct_seqs += (correct_tokens == num_tokens).sum()

            train_loss /= batch + 1
            seq_acc = correct_seqs / val_size
            token_acc /= val_size
            if epoch == 0:
                print('+' * 64)
            print('Epoch {:d}'.format(epoch + 1))
            print('-' * 64)
            print('Train | Loss | {:.6f}'.format(train_loss))
            print('Val | Seq Acc. | {:.4f} | Token Acc. | {:.4f}'.format(
                seq_acc, token_acc))
            print('-' * 64)
            # Check whether early stops
            stop_dectector(seq_acc, token_acc, model)
            if stop_dectector.early_stop:
                print('+' * 64)
                print('Training early stops at {:d}'.format(epoch + 1))
                break
            print('+' * 64)

        if not stop_dectector.early_stop:
            # Save model at the last epoch if even token accuracy remains zero
            if not exists(model_path):
                save(model.state_dict(), model_path)
                print(
                    'The model at the last epoch is saved in \'{:s}\'.'.format(model_path))
            print('Training process normally stops.')

    def inference(self, model: Transformer, data: DataLoader, vocab: List, max_seq_len: Union[List, int], contains_spec:bool=False) -> None:
        r'''Inference during validation or prediction.'''
        total = int(self.dataset_size * (0.1 if self.valid else 0.2))
        device = next(model.parameters()).device
        model.eval()
        if not getattr(self, 'collect', False) and getattr(self, 'accum', False):

            if getattr(self, 'pe_scheme_accum', '') == 'pc':
                TA = SE = ME = nc1 = nc2 = 0
                for prompt, prompt_mask, prompt_id1, prompt_id2, response, _, _, _, _ in data:
                    b = prompt.size(0)
                    prompt = cat([prompt, ones(b, 1).long()], dim=1)
                    mask = cat([prompt_mask, zeros(b, 1).bool()], dim=1)
                    prompt_id1 = cat([prompt_id1, zeros(b, 1).long()], dim=1)
                    prompt_id2 = cat([prompt_id2, ones(b, 1).long()], dim=1)
                    targets = response[:, 1:]
                    nid1 = zeros(b).long().to(device)
                    nid2 = ones(b).long().to(device)
                    for i in range(max_seq_len):
                        if i:
                            with no_grad():
                                nid = model(x=nid.to(device), x_mask=mask.to(device), posid1=tensor(
                                    nid1).to(device), posid2=tensor(nid2).to(device)).argmax(dim=-1)
                                predicts = cat(
                                    [predicts, nid.unsqueeze(1)], dim=1)
                        else:
                            with no_grad():
                                nid = model(x=prompt.to(device), x_mask=mask.to(device), posid1=prompt_id1.to(
                                    device), posid2=prompt_id2.to(device), max_seq_len=max_seq_len).argmax(dim=-1)
                            predicts = nid.unsqueeze(1)
                        if (predicts == 2).any(dim=1).sum() == b:
                            break
                        # To ensure PosID1 of unit digit is 2
                        nid1[where(nid1 == 0)[0]] = 1
                        nid1 += 1  # Digit left shift
                        nid1[where((nid == 4) | (nid == 5))[0]] = 1  # '+', '-'
                        # Next operand when encountering '>'
                        nid1[where((nid < 4) | (nid1 > NUM_DIGITS + 1))[0]] = 0
                        nid2 += nid == 3
                        nid2[where((nid < 3) | (nid2 > NUM_OPERANDS))[0]] = 0
                        mask = cat([mask, zeros(b, 1).bool()], dim=1)
                    for i in range(b):
                        p_tokens, predict = ids_to_str(
                            predicts[i], vocab, True)
                        t_tokens, target = ids_to_str(targets[i], vocab, True)
                        p_len = len(p_tokens)
                        if p_len == len(t_tokens):
                            TA += sum(1 for j in range(p_len)
                                      if p_tokens[j] == t_tokens[j]) / p_len
                        _SE = predict == target
                        res_p, res_t = fetch_res_in_scp(
                            predict)[::-1], fetch_res_in_scp(target)[::-1]
                        if res_p == res_t == '0+':
                            _ME = True
                        else:
                            try:
                                _ME = Int32X(res_p) == Int32X(res_t)
                            except:
                                _ME = False
                        SE, ME = SE + _SE, ME + _ME
                        if not _ME and nc1 < 5:
                            print('-' * 64)
                            print('Wrong result')
                            print('Input:\n{:s}'.format(
                                ids_to_str(prompt[i, 1:-1], vocab)))
                            print('Target:\n{:s}'.format(target))
                            print('Prediction:\n{:s}'.format(predict))
                            nc1 += 1
                        if not _SE and _ME and nc2 < 5:
                            print('-' * 64)
                            print('Correct result but different scratchpad')
                            print('Input:\n{:s}'.format(
                                ids_to_str(prompt[i, 1:-1], vocab)))
                            print('Target:\n{:s}'.format(target))
                            print('Prediction:\n{:s}'.format(predict))
                            nc2 += 1
                print('+' * 64)
                print('Metrics')
                print('+' * 64)
                print('Token Acc.: {:.4f}'.format(TA / total * 100))
                print('Seq Acc.: {:.4f}'.format(SE / total * 100))
                print('Math Acc.: {:.4f}'.format(ME / total * 100))

            else:
                '''ToDo: decoder-only Transformer with other PEs.'''
                pass

        else:
            contain_collected_coeffs = (getattr(self, 'collect', False) or getattr(self, 'sum', False)) and not getattr(
                self, 'combine', False) and not getattr(self, 'accum', False)
            copilot = InferenceResults(
                total, vocab, self.task, self.task == 'mul' and self.dataset == 'v1d7C1', contain_collected_coeffs)
            for in_vocabid, in_mask, out_vocabid, _ in data:
                b = in_vocabid.size(0)
                in_mask = in_mask.to(device)
                for i in range(max_seq_len[1] - 1):
                    if i:
                        with no_grad():
                            nid = model(nid, in_mask=in_mask).argmax(dim=-1)
                        predicts = cat(
                            [predicts, nid.unsqueeze(1).cpu()], dim=1)
                    else:
                        with no_grad():
                            nid = model(ones(b).long().to(device), None, in_vocabid.to(
                                device), in_mask, max_seq_len=max_seq_len).argmax(dim=-1)
                        predicts = nid.unsqueeze(1).cpu()
                    if (predicts == 2).any(dim=1).sum() == b:
                        break
                for i in range(b):
                    predict = predicts[i]
                    target = out_vocabid[i]
                    if contains_spec:
                        predict, _ = ids_to_str(predict, vocab, True)
                        predict = fetch_res_in_scp(predict)
                        target, _ = ids_to_str(target, vocab, True)
                        target = fetch_res_in_scp(target)
                    copilot.iter(in_vocabid[i], target, predict)
            copilot.report()

    def collect_and_combine(self, model_collect: Transformer, model_combine: Transformer, data: DataLoader) -> None:
        r'''Conduct polynomial multiplication/addition via learning collected products/sums and then combining them.'''
        device = next(model_collect.parameters()).device
        total = int(self.dataset_size * (0.1 if self.valid else 0.2))
        copilot = InferenceResults(total, self.vocab, 'mul')
        model_collect.eval()
        model_combine.eval()
        for in_vocabid, in_mask, out_vocabid, _ in data:
            # Subtask of learning collected product
            b = in_vocabid.size(0)
            # From `vocab` to `vocab_accum` (extra '(' & ')' after '<pad>', '<s>', '</s>', '<mul>')
            in_vocabid_collect = in_vocabid + (in_vocabid > 3) * 2
            in_mask = in_mask.to(device)
            for i in range(self.max_out_seq_len_collect):
                if i:
                    with no_grad():
                        nid_collect = model_collect(
                            nid_collect, in_mask=in_mask).argmax(dim=-1)
                    predicts_collect = cat(
                        [predicts_collect, nid_collect.unsqueeze(1).cpu()], dim=1)
                else:
                    with no_grad():
                        nid_collect = model_collect(ones(b).long().to(device), None, in_vocabid_collect.to(
                            device), in_mask, max_seq_len=[self.max_in_seq_len_collect, self.max_out_seq_len_collect]).argmax(dim=-1)
                    predicts_collect = nid_collect.unsqueeze(1).cpu()
                if (predicts_collect == 2).any(dim=1).sum() == b:
                    break
            in_vocabid_combine = cat(
                [ones(b, 1).long(), predicts_collect], dim=1)
            in_mask = arange(in_vocabid_combine.size(1)).unsqueeze(0) > (
                in_vocabid_combine == 2).int().argmax(dim=1, keepdim=True)
            in_mask = in_mask.to(device)
            for i in range(self.max_out_seq_len_combine):
                if i:
                    with no_grad():
                        nid_combine = model_combine(
                            nid_combine, in_mask=in_mask).argmax(dim=-1)
                    predicts = cat(
                        [predicts, nid_combine.unsqueeze(1).cpu()], dim=1)
                else:
                    with no_grad():
                        nid_combine = model_combine(ones(b).long().to(device), None, in_vocabid_combine.to(
                            device), in_mask, max_seq_len=[self.max_in_seq_len_combine, self.max_out_seq_len_combine]).argmax(dim=-1)
                    predicts = nid_combine.unsqueeze(1).cpu()
                if (predicts == 2).any(dim=1).sum() == b:
                    break
            for i in range(b):
                p_tokens, _ = ids_to_str(
                    predicts[i], self.vocab_collect, True)
                copilot.collect_output = ''.join(ids_to_str(predicts_collect[i], self.vocab_collect))
                copilot.iter(in_vocabid[i], out_vocabid[i], p_tokens)
        copilot.report()

    def collect_and_accum(self, model_collect: Transformer, model_accum: Transformer, data: DataLoader) -> None:
        r'''Conduct polynomial multiplication via learning collected products and accumulate collected coefficients.'''
        device = next(model_collect.parameters()).device
        total = int(self.dataset_size * (0.1 if self.valid else 0.2))
        copilot = InferenceResults(total, self.vocab, 'mul')
        model_collect.eval()
        model_accum.eval()
        for in_vocabid, in_mask, out_vocabid, _ in data:
            # Subtask of learning collected product
            b = in_vocabid.size(0)
            # From `vocab` to `vocab_accum` (extra '(' & ')' after '<pad>', '<s>', '</s>', '<mul>')
            in_vocabid_collect = in_vocabid + (in_vocabid > 3) * 2
            in_mask = in_mask.to(device)
            for i in range(self.max_out_seq_len_collect - 1):
                if i:
                    with no_grad():
                        nid_collect = model_collect(
                            nid_collect, in_mask=in_mask).argmax(dim=-1)
                    predicts_collect = cat(
                        [predicts_collect, nid_collect.unsqueeze(1).cpu()], dim=1)
                else:
                    with no_grad():
                        nid_collect = model_collect(ones(b).long().to(device), None, in_vocabid_collect.to(
                            device), in_mask, max_seq_len=[self.max_in_seq_len_collect, self.max_out_seq_len_collect]).argmax(dim=-1)
                    predicts_collect = nid_collect.unsqueeze(1).cpu()
                if (predicts_collect == 2).any(dim=1).sum() == b:
                    break
            predicts_collect = predicts_collect.to('cpu')
            replacement, prompts, prompt_id1s, prompt_id2s = separate_collected_coeffs(
                predicts_collect, self.vocab_collect, self.vocab_accum)
            # Accumulation subtask
            combined_coeffs = []
            for prompt, mask, prompt_id1, prompt_id2 in accum_list_dataloader(prompts, prompt_id1s, prompt_id2s, self.batch_size):
                b_accum = prompt.size(0)
                nid1 = zeros(b_accum).long().to(device)
                nid2 = ones(b_accum).long().to(device)
                for i in range(self.max_seq_len_accum - prompt.size(1)):
                    if i:
                        with no_grad():
                            nid_accum = model_accum(x=nid_accum.to(device), x_mask=mask.to(device), posid1=tensor(
                                nid1).to(device), posid2=tensor(nid2).to(device)).argmax(dim=-1)
                            predicts_accum = cat(
                                [predicts_accum, nid_accum.unsqueeze(1)], dim=1)
                    else:
                        with no_grad():
                            nid_accum = model_accum(x=prompt.to(device), x_mask=mask.to(device), posid1=prompt_id1.to(
                                device), posid2=prompt_id2.to(device), max_seq_len=self.max_seq_len_accum).argmax(dim=-1)
                        predicts_accum = nid_accum.unsqueeze(1)
                    if (predicts_accum == 2).any(dim=1).sum() == b_accum:
                        break
                    nid1[where(nid1 == 0)[0]] = 1
                    nid1 += 1
                    nid1[where((nid_accum == 4) | (nid_accum == 5))[0]] = 1
                    nid1[where((nid_accum < 4) | (
                        nid1 > NUM_DIGITS + 1))[0]] = 0
                    nid2 += nid_accum == 3
                    nid2[where((nid_accum < 3) | (nid2 > NUM_OPERANDS))[0]] = 0
                    mask = cat([mask, zeros(b_accum, 1).bool()], dim=1)
                predicts_accum = predicts_accum.to('cpu')
                for i in range(b_accum):
                    combined_coeffs.append(fetch_res_in_scp(
                        ids_to_str(predicts_accum[i], self.vocab_accum))[::-1])
            # Replace collected coefficients with combined coefficient
            for i in range(b):
                collect_tokens, collect_str = ids_to_str(
                    predicts_collect[i], self.vocab_collect, True)
                copilot.collect_output = collect_str
                p_tokens = collect_tokens
                replaces = tensor(replacement[i])
                for rid in unique(replaces[replaces > 0]).sort(descending=True).values:
                    collect_idx = (replaces == rid).nonzero(as_tuple=True)[0]
                    lbound, rbound = collect_idx.min(), collect_idx.max()
                    p_tokens[lbound - 1: rbound +
                             2] = list(combined_coeffs[rid - 1])
                copilot.iter(in_vocabid[i], out_vocabid[i], p_tokens)
        copilot.report()

    def collect_and_accum_submodule(self, collect_model: Transformer, model_accum: Transformer, in_vocabid_collect: Tensor, in_vocabid_collect_mask: Tensor, b: int, device: str, max_collect_len) -> Tensor:
        in_vocabid_collect_mask = in_vocabid_collect_mask.to(device)
        for i in range(max_collect_len[1] - 1):
            if i:
                with no_grad():
                    nid_collect = collect_model(
                        nid_collect, in_mask=in_vocabid_collect_mask).argmax(dim=-1)
                predicts_collect = cat(
                    [predicts_collect, nid_collect.unsqueeze(1).cpu()], dim=1)
            else:
                with no_grad():
                    nid_collect = collect_model(ones(b).long().to(device), None, in_vocabid_collect.to(
                        device), in_vocabid_collect_mask, max_seq_len=[max_collect_len[0], max_collect_len[1]]).argmax(dim=-1)
                predicts_collect = nid_collect.unsqueeze(1).cpu()
            if (predicts_collect == 2).any(dim=1).sum() == b:
                break
        predicts_collect = predicts_collect.to('cpu')
        replacement, prompts, prompt_id1s, prompt_id2s = separate_collected_coeffs(
            predicts_collect, self.vocab, self.vocab_accum)
        # Accumulation subtask
        combined_coeffs = []
        for prompt, mask, prompt_id1, prompt_id2 in accum_list_dataloader(prompts, prompt_id1s, prompt_id2s, self.batch_size):
            b_accum = prompt.size(0)
            nid1 = zeros(b_accum).long().to(device)
            nid2 = ones(b_accum).long().to(device)
            for i in range(self.max_seq_len_accum - prompt.size(1)):
                if i:
                    with no_grad():
                        nid_accum = model_accum(x=nid_accum.to(device), x_mask=mask.to(device), posid1=tensor(
                            nid1).to(device), posid2=tensor(nid2).to(device)).argmax(dim=-1)
                        predicts_accum = cat(
                            [predicts_accum, nid_accum.unsqueeze(1)], dim=1)
                else:
                    with no_grad():
                        nid_accum = model_accum(x=prompt.to(device), x_mask=mask.to(device), posid1=prompt_id1.to(
                            device), posid2=prompt_id2.to(device), max_seq_len=self.max_seq_len_accum).argmax(dim=-1)
                    predicts_accum = nid_accum.unsqueeze(1)
                if (predicts_accum == 2).any(dim=1).sum() == b_accum:
                    break
                nid1[where(nid1 == 0)[0]] = 1
                nid1 += 1
                nid1[where((nid_accum == 4) | (nid_accum == 5))[0]] = 1
                nid1[where((nid_accum < 4) | (
                    nid1 > NUM_DIGITS + 1))[0]] = 0
                nid2 += nid_accum == 3
                nid2[where((nid_accum < 3) | (nid2 > NUM_OPERANDS))[0]] = 0
                mask = cat([mask, zeros(b_accum, 1).bool()], dim=1)
            predicts_accum = predicts_accum.to('cpu')
            for i in range(b_accum):
                combined_coeffs.append(fetch_res_in_scp(
                    ids_to_str(predicts_accum[i], self.vocab_accum))[::-1])
        # Replace collected coefficients with combined coefficient
        tensor_list = []
        for i in range(b):
            p_tokens, _ = ids_to_str(
                predicts_collect[i], self.vocab, True)
            replaces = tensor(replacement[i])
            for rid in unique(replaces[replaces > 0]).sort(descending=True).values:
                collect_idx = (replaces == rid).nonzero(as_tuple=True)[0]
                lbound, rbound = collect_idx.min(), collect_idx.max()
                p_tokens[lbound - 1: rbound +
                         2] = list(combined_coeffs[rid - 1])
            tensor_list.append(
                tensor([self.vocab.index(token) for token in p_tokens]))
        return tensor_list

    def long_seq_mul(self, model_collect: Transformer, model_accum: Transformer, model_sum: Transformer, data: DataLoader) -> None:
        device = next(model_collect.parameters()).device
        total = int(self.dataset_size * (0.1 if self.valid else 0.2))
        copilot = InferenceResults(total, self.vocab, 'mul')
        model_collect.eval()
        model_accum.eval()
        model_sum.eval()
        for f0g0, f0g0_mask, f0g1, f0g1_mask, f1g0, f1g0_mask, f1g1, f1g1_mask, product, _ in data:
            b = f0g0.size(0)
            p00 = self.collect_and_accum_submodule(model_collect, model_accum, f0g0, f0g0_mask, b, device, [
                                                   self.max_in_seq_len_collect, self.max_out_seq_len_collect])
            p01 = self.collect_and_accum_submodule(model_collect, model_accum, f0g1, f0g1_mask, b, device, [
                                                   self.max_in_seq_len_collect, self.max_out_seq_len_collect])
            p10 = self.collect_and_accum_submodule(model_collect, model_accum, f1g0, f1g0_mask, b, device, [
                                                   self.max_in_seq_len_collect, self.max_out_seq_len_collect])
            p11 = self.collect_and_accum_submodule(model_collect, model_accum, f1g1, f1g1_mask, b, device, [
                                                   self.max_in_seq_len_collect, self.max_out_seq_len_collect])

            p01_p10 = pad_sequence([cat([tensor([1]), t1, tensor(
                [3]), t2, tensor([2])], dim=0) for t1, t2 in zip(p01, p10)], True)
            p01_p10_mask = p01_p10 == 0
            s1 = self.collect_and_accum_submodule(model_sum, model_accum, p01_p10, p01_p10_mask, b, device, [
                                                  self.max_in_seq_len_sum, self.max_out_seq_len_sum])
            s1_p11 = pad_sequence([cat([tensor([1]), t1, tensor(
                [3]), t2, tensor([2])], dim=0) for t1, t2 in zip(s1, p11)], True)
            s1_p11_mask = s1_p11 == 0
            s1_s2 = self.collect_and_accum_submodule(model_sum, model_accum, s1_p11, s1_p11_mask, b, device, [
                                                     self.max_in_seq_len_sum, self.max_out_seq_len_sum])
            p00_s1_s2 = pad_sequence([cat([tensor([1]), t1, tensor(
                [3]), t2, tensor([2])], dim=0) for t1, t2 in zip(p00, s1_s2)], True)
            p00_s1_s2_mask = p00_s1_s2 == 0
            predicts = self.collect_and_accum_submodule(model_sum, model_accum, p00_s1_s2, p00_s1_s2_mask, b, device, [
                                                        self.max_in_seq_len_sum, self.max_out_seq_len_sum])
            for i in range(b):
                sep00 = (f0g0[i] == 3).nonzero(as_tuple=True)[0]
                f0, g0 = f0g0[i][:sep00], f0g0[i][sep00 + 1:-1]
                sep11 = (f1g1[i] == 3).nonzero(as_tuple=True)[0]
                f1, g1 = f1g1[i][1:sep11], f1g1[i][sep11 + 1:]
                in_vocabid = cat([f0, f1, tensor([3]), g0, g1], dim=0)
                copilot.iter(in_vocabid, product[i], predicts[i])
        copilot.report()
        
    def train_polynormer(self, model: Polynormer, train_loader: DataLoader, val_loader: DataLoader, model_path: str) -> None:
        r'''Training the model.'''
        seed_everything(0)
        val_size = int(self.dataset_size * 0.1)

        optimizer = Ranger(model.parameters(), lr=self.lr)
        loss_fn = CrossEntropyLoss(ignore_index=0)  # Ignore paddings
        stop_dectector = EarlyStopping(self.patience, self.lod, model_path)

        for epoch in range(self.num_epochs):
            train_loss = 0
            correct_seqs = 0
            token_acc = 0
            model.train()
            for batch, (in_graph, out_vocabid, out_mask) in enumerate(train_loader):
                out_vocabid = out_vocabid.to(
                        next(model.parameters()).device)
                out_mask = out_mask[:, :-
                                        1].to(next(model.parameters()).device)
                logits = model(
                        out_vocabid[:, :-1], out_mask, in_graph, teacher_forcing=True)
                loss = loss_fn(logits.flatten(0, 1),
                                   out_vocabid[:, 1:].flatten())
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                train_loss += loss.item()
            model.eval()
            for in_graph, out_vocabid, out_mask in val_loader:
                in_vocabid = in_vocabid.to(next(model.parameters()).device)
                in_mask = in_mask.to(next(model.parameters()).device)
                out_vocabid = out_vocabid.to(
                        next(model.parameters()).device)
                out_mask = out_mask[:, :-
                                        1].to(next(model.parameters()).device)
                with no_grad():
                    logits = model(
                            out_vocabid[:, :-1], out_mask, in_vocabid, in_mask, teacher_forcing=True)
                predicts = logits.argmax(dim=-1)
                targets = (out_vocabid * out_vocabid.sgn())[:, 1:]
                num_tokens = (targets != 0).sum(dim=1)
                correct_tokens = ((predicts == targets) &
                                      (targets != 0)).sum(dim=1)
                token_acc += (correct_tokens / num_tokens).sum()
                correct_seqs += (correct_tokens == num_tokens).sum()

            train_loss /= batch + 1
            seq_acc = correct_seqs / val_size
            token_acc /= val_size
            if epoch == 0:
                print('+' * 64)
            print('Epoch {:d}'.format(epoch + 1))
            print('-' * 64)
            print('Train | Loss | {:.6f}'.format(train_loss))
            print('Val | Seq Acc. | {:.4f} | Token Acc. | {:.4f}'.format(
                seq_acc, token_acc))
            print('-' * 64)
            # Check whether early stops
            stop_dectector(seq_acc, token_acc, model)
            if stop_dectector.early_stop:
                print('+' * 64)
                print('Training early stops at {:d}'.format(epoch + 1))
                break
            print('+' * 64)

        if not stop_dectector.early_stop:
            # Save model at the last epoch if even token accuracy remains zero
            if not exists(model_path):
                save(model.state_dict(), model_path)
                print(
                    'The model at the last epoch is saved in \'{:s}\'.'.format(model_path))
            print('Training process normally stops.')

    def run(self, **kwargs) -> None:
        r'''Conducts training or inference.'''
        device = 'cuda' if cuda.is_available() else 'cpu'
        # device = 'cpu'
        generator = PolyDataloader(**kwargs)

        if getattr(self, 'collect', False) + getattr(self, 'combine', False) + getattr(self, 'accum', False) + getattr(self, 'sum', False):
            if getattr(self, 'collect', False):
                model_collect = Transformer(self.pe_scheme_collect, len(self.vocab_collect), self.emb_dim_collect, self.num_heads_collect, self.num_ffn_layers_collect, self.hidden_dim_collect,
                                            self.activate_fn_collect, self.norm_scheme_collect, self.drop_prob, self.num_encoder_layers_collect, self.num_decoder_layers_collect).to(device)
            if getattr(self, 'combine', False):
                model_combine = Transformer(self.pe_scheme_combine, len(self.vocab_collect), self.emb_dim_combine, self.num_heads_combine, self.num_ffn_layers_combine, self.hidden_dim_combine,
                                            self.activate_fn_combine, self.norm_scheme_combine, self.drop_prob, self.num_encoder_layers_combine, self.num_decoder_layers_combine).to(device)
            if self.accum:
                model_accum = Transformer(self.pe_scheme_accum, len(self.vocab_accum), self.emb_dim_accum, self.num_heads_accum, self.num_ffn_layers_accum,
                                          self.hidden_dim_accum, self.activate_fn_accum, self.norm_scheme_accum, self.drop_prob, 0, self.num_decoder_layers_accum).to(device)
            if self.sum:
                model_sum = Transformer(self.pe_scheme_sum, len(self.vocab), self.emb_dim_sum, self.num_heads_sum, self.num_ffn_layers_sum,
                                        self.hidden_dim_sum, self.activate_fn_sum, self.norm_scheme_sum, self.drop_prob, self.num_encoder_layers_sum, self.num_decoder_layers_sum).to(device)
        else:
            model = Transformer(self.pe_scheme, len(self.vocab), self.emb_dim, self.num_heads, self.num_ffn_layers, self.hidden_dim,
                                self.activate_fn, self.norm_scheme, self.drop_prob, self.num_encoder_layers, self.num_decoder_layers).to(device)

        if self.train:
            train_loader, val_loader = generator.construct()
            print('Training Transformer model ...')
            if getattr(self, 'collect', False) + getattr(self, 'combine', False) + getattr(self, 'accum', False) + getattr(self, 'sum', False):
                if getattr(self, 'collect', False):
                    self.training(model_collect, train_loader,
                                  val_loader, self.collect_model_path)
                elif getattr(self, 'combine', False):
                    self.training(model_combine, train_loader,
                                  val_loader, self.combine_model_path)
                elif self.accum:
                    self.training(model_accum, train_loader,
                                  val_loader, self.accum_model_path)
                else:
                    self.training(model_sum, train_loader,
                                  val_loader, self.sum_model_path)
            else:
                if self.finetune:
                    if self.new_torch:
                        pretrained_dict= load(self.ft_model_path, weights_only=False)
                    else:
                        pretrained_dict= load(self.ft_model_path)
                    if self.dataset[:2] == 'v3':
                        if self.dataset in ['v3t3e2C1L3', 'v3t3e2F7']:
                            emb = pretrained_dict.pop('word_emb.emb.weight')
                            proj = pretrained_dict.pop('out_proj.weight')
                        if self.dataset in ['v3t3E1F7', 'v3t3e2F7']:
                            ape_in = pretrained_dict.pop('pos_emb.ape_in')
                            ape_out = pretrained_dict.pop('pos_emb.ape_out')
                        model.load_state_dict(pretrained_dict, strict=False)
                        if self.dataset in ['v3t3e2C1L3', 'v3t3e2F7']:
                            with no_grad():
                                new_weight = model.word_emb.emb.weight.detach().clone()
                                new_weight[:emb.size(0)] = emb
                                model.word_emb.emb.weight.data.copy_(new_weight)
                                new_weight = model.out_proj.weight.detach().clone()
                                new_weight[:proj.size(0)] = proj
                                model.word_emb.emb.weight.data.copy_(new_weight)
                        if self.dataset in ['v3t3E1F7', 'v3t3e2F7']:
                            with no_grad():
                                new_ape_in = model.pos_emb.ape_in.detach().clone()
                                new_ape_in[:ape_in.size(0)] = ape_in
                                model.pos_emb.ape_in.data.copy_(new_ape_in)
                                new_ape_out = model.pos_emb.ape_out.detach().clone()
                                new_ape_out[:ape_out.size(0)] = ape_out
                                model.pos_emb.ape_out.data.copy_(new_ape_out)
                    else:
                        model.load_state_dict(pretrained_dict)
                    
                    # Freeze the former half encoder layers
                    if not self.no_freeze:
                        for i in range(self.num_encoder_layers // 2):
                            for param in model.encoder[i].parameters():
                                param.requires_grad = False
                if self.model == 'polynormer':
                    self.train_polynormer(model, train_loader, val_loader, self.model_path)
                else:
                    self.training(model, train_loader, val_loader, self.model_path)
        else:
            inference_data = generator.construct()
            print(('Validating' if self.valid else 'Testing') +
                  ' Transformer model ...')
            if self.model == 'polynormer':
                pass
            if getattr(self, 'collect', False):
                if self.new_torch:
                    model_collect.load_state_dict(
                        load(self.collect_model_path, weights_only=False))
                else:
                    model_collect.load_state_dict(
                        load(self.collect_model_path))
                print('Model for the subtask of learning collected product is loaded from \'{:s}\'.'.format(
                    self.collect_model_path))
            if getattr(self, 'combine', False):
                if self.new_torch:
                    model_combine.load_state_dict(
                        load(self.combine_model_path, weights_only=False))
                else:
                    model_combine.load_state_dict(
                        load(self.combine_model_path))
                print('Model for the subtask of combining collected product is loaded from \'{:s}\'.'.format(
                    self.combine_model_path))
            if getattr(self, 'accum', False):
                if self.new_torch:
                    model_accum.load_state_dict(
                        load(self.accum_model_path, weights_only=False))
                else:
                    model_accum.load_state_dict(load(self.accum_model_path))
                print('Model for integer accumulation is loaded from \'{:s}\'.'.format(
                    self.accum_model_path))
            if getattr(self, 'sum', False):
                if self.new_torch:
                    model_sum.load_state_dict(
                        load(self.sum_model_path, weights_only=False))
                else:
                    model_sum.load_state_dict(load(self.sum_model_path))
                print('Model for the subtask of learning collected sum is loaded from \'{:s}\'.'.format(
                    self.sum_model_path))
            if getattr(self, 'collect', False) + getattr(self, 'combine', False) + getattr(self, 'accum', False) + getattr(self, 'sum', False) == 1:
                if getattr(self, 'collect', False):
                    self.inference(model_collect, inference_data, self.vocab_collect, [
                                   self.max_in_seq_len_collect, self.max_out_seq_len_collect])
                elif getattr(self, 'combine', False):
                    self.inference(model_combine, inference_data, self.vocab_collect, [
                                   self.max_in_seq_len_combine, self.max_out_seq_len_combine])
                elif self.accum:
                    self.inference(model_accum, inference_data,
                                   self.vocab_accum, self.max_seq_len_accum)
                else:
                    self.inference(model_sum, inference_data,
                                   self.vocab, [self.max_in_seq_len_sum, self.max_out_seq_len_sum])
            elif getattr(self, 'collect', False):
                if self.combine:
                    self.collect_and_combine(
                        model_collect, model_combine, inference_data)
                elif self.sum:
                    self.long_seq_mul(model_collect, model_accum,
                                      model_sum, inference_data)
                else:
                    self.collect_and_accum(
                        model_collect, model_accum, inference_data)
            elif getattr(self, 'accum', False) and getattr(self, 'sum', False):
                self.collect_and_accum(model_sum, model_accum, inference_data)
            else:
                if self.new_torch:
                    model.load_state_dict(
                        load(self.model_path, weights_only=False))
                else:
                    model.load_state_dict(load(self.model_path))
                contains_spec = False
                if self.task == 'prem':
                    if self.prem_step == '0':
                        if self.detailed_spa:
                            contains_spec = True
                    else:
                        contains_spec = True
                self.inference(model, inference_data, self.vocab, [
                               self.max_in_seq_len, self.max_out_seq_len], contains_spec)

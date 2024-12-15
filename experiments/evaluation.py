import torch
from experiments.metric import Flags, Flag
from copy import deepcopy
from collections import OrderedDict

def predict_batch(model, src_sequence, vocab, max_len=128, device="cpu"):
    """
    Generates predictions for a batch of source sequences using the given model.
    Args:
        model (torch.nn.Module): The model used for generating predictions.
        src_sequence (torch.Tensor): The source sequences to be translated.
        vocab (Vocabulary): The vocabulary object containing all tokens.
        max_len (int, optional): The maximum length of the generated sequences. Defaults to 128.
        device (str, optional): The device to run the model on ("cpu" or "cuda"). Defaults to "cpu".
    Returns:
        torch.Tensor: The generated target sequences.
    """
    model.eval()
    with torch.no_grad():
        # Prepare source sequence
        src = src_sequence.to(device) 
        src_mask = model.create_src_mask(src)

        # Generate initial encoder output
        encoder_output = model.encoder(src, src_mask)

        # Initialize target sequence with SOS token
        tgt = [vocab.sos_idx] + [vocab.pad_idx] * (max_len)
        tgt = torch.tensor([tgt], device=device)
        tgt = tgt.repeat(src.size(0), 1)

        # holds indx of sequences containing EOS token
        finished = torch.tensor([False]*src.size(0), device=device)
        
        # Generate tokens one by one
        for i in range(1,max_len+1):
            # indx of batch dim where EOS token has not been generated
            active_indxs = torch.where(finished == False)[0]
            # feed only unfinished sequences to decoder
            tgt_active = tgt[active_indxs]
            # remove padding
            tgt_active = tgt_active[:,0:i]
            # create mask
            tgt_mask = model.create_tgt_mask(tgt_active)
            # decode
            decoder_output = model.decoder(tgt_active,
                                           encoder_output[active_indxs],
                                           src_mask[active_indxs],
                                           tgt_mask)
            # Get next token prediction
            next_token = decoder_output.argmax(dim=-1)
            next_token = next_token[:, -1]
            # store new prediction for active sequences
            tgt[active_indxs,i] = next_token
            
            # update finished sequences if any EOS token is generated
            new_finished = torch.where(next_token == vocab.eos_idx, True, False)
            finished[active_indxs] = torch.logical_or(finished[active_indxs], new_finished)

            # early stopping if all sequences produced EOS token
            if finished.all():
                break
    return tgt

def predict_batch_oracle(model, src_sequence, vocab, seq_lenghts, device="cpu"):
    """
    Predicts a batch of sequences using the given model and source sequences.
    Args:
        model (torch.nn.Module): The model to use for prediction.
        src_sequence (torch.Tensor): The source sequences to predict from.
        vocab (Vocabulary): The vocabulary object containing all tokens.
        seq_lenghts (list[int]): The lengths of the sequences to predict.
        device (str, optional): The device to run the prediction on. Defaults to "cpu".
    Returns:
        torch.Tensor: The predicted target sequences.
    """

    model.eval()
    with torch.no_grad():
        # Prepare source sequence
        src = src_sequence.to(device) 
        src_mask = model.create_src_mask(src)

        # Generate initial encoder output
        encoder_output = model.encoder(src, src_mask)
        #seq_lenghts += 1
        longest = max(seq_lenghts)
        # Initialize target sequence with SOS token
        tgt = [vocab.sos_idx] + [vocab.pad_idx] * (longest)
        tgt = torch.tensor([tgt], device=device)
        tgt = tgt.repeat(src.size(0), 1)

        # holds indx of sequences containing with target length
        finished = torch.tensor([False]*src.size(0), device=device)
        
        # Generate tokens one by one
        for i in range(1,longest):
            # indx of batch dim where seq len is not yet matched
            active_indxs = torch.where(finished == False)[0]
            # feed only unfinished sequences to decoder
            tgt_active = tgt[active_indxs]
            # remove padding
            tgt_active = tgt_active[:,0:i]
            # create mask
            tgt_mask = model.create_tgt_mask(tgt_active)
            # decode
            decoder_output = model.decoder(tgt_active,
                                           encoder_output[active_indxs],
                                           src_mask[active_indxs],
                                           tgt_mask)
            # consider only prediction for last token
            decoder_output = decoder_output[:,-1,:]
            # Get next token prediction
            next_token = decoder_output.argmax(dim=-1)
            # seqences with target length
            new_finished = torch.where(seq_lenghts[active_indxs] == (i), True, False)
            # if EOS token is predicted, use second best prediction
            early_eos = torch.where((next_token == vocab.eos_idx),True, False)
            if early_eos.any():
                early_eos_indx = next_token[early_eos]
                decoder_output[:,early_eos_indx] = 0
                next_token = decoder_output.argmax(dim=-1)

            # store new prediction for active sequences
            tgt[active_indxs,i] = next_token
            # update finished sequences 
            finished[active_indxs] = torch.logical_or(finished[active_indxs], new_finished)

            # early stopping if all sequences produced EOS token
            if finished.all():
                break
    return tgt

def _dummy_eval_result():
    inp_len_max = 10
    tgt_len_max = 10
    inp_seq_span = torch.zeros(inp_len_max+1, device="cpu", dtype=torch.long)
    tgt_seq_span = torch.zeros(tgt_len_max+1, device="cpu", dtype=torch.long)

    template = OrderedDict({
        Flags.DistrFlags.Avrg:
            OrderedDict({
                Flags.LevelFlags.TL: 0,
                Flags.LevelFlags.SL: 0,
            }),
        Flags.DistrFlags.InLen:
            OrderedDict({
                Flags.LevelFlags.TL: inp_seq_span.clone(),
                Flags.LevelFlags.SL: inp_seq_span.clone(),
            }),
        Flags.DistrFlags.OutLen:
            OrderedDict({
                Flags.LevelFlags.TL: tgt_seq_span.clone(),
                Flags.LevelFlags.SL: tgt_seq_span.clone(),
            })
    })
    correct = deepcopy(template)
    total = deepcopy(template)
    err_rate = deepcopy(template)
    accuracy = deepcopy(template)

    result = OrderedDict({
        Flags.MetricFlags.CORRECT : detach_result_map(correct),
        Flags.MetricFlags.TOTAL : detach_result_map(total),
        Flags.MetricFlags.ACC : detach_result_map(accuracy),
        Flags.MetricFlags.ERR : detach_result_map(err_rate)
    })

    ret = OrderedDict()
    ret[Flags.PredictionFlags.ORACLE] = deepcopy(result)
    ret[Flags.PredictionFlags.NO_ORACLE] = result    
    ret = {Flags.PredictionFlags.ORACLE: deepcopy(result),
                Flags.PredictionFlags.NO_ORACLE: result}
    
    
    return ret
    
def _evaluate_model_batchwise(model, test_data, test_loader, vocab, device="cpu", length_oracle=False):
    inp_len_max = test_data.command_length_span[1]
    tgt_len_max = test_data.action_sequence_span[1]
    inp_seq_span = torch.zeros(inp_len_max+1, device=device, dtype=torch.long)
    tgt_seq_span = torch.zeros(tgt_len_max+1, device=device, dtype=torch.long)

    template = {
        Flags.DistrFlags.Avrg:
            {
                Flags.LevelFlags.TL: 0,
                Flags.LevelFlags.SL: 0,
            },
        Flags.DistrFlags.InLen:
            {
                Flags.LevelFlags.TL: inp_seq_span.clone(),
                Flags.LevelFlags.SL: inp_seq_span.clone(),
            },
        Flags.DistrFlags.OutLen:
            {
                Flags.LevelFlags.TL: tgt_seq_span.clone(),
                Flags.LevelFlags.SL: tgt_seq_span.clone(),
            }
    }
    correct = deepcopy(template)
    total = deepcopy(template)


    for src, _, tgt in test_loader:
        src = src.to(device)
        is_special_F = lambda x: (x == vocab.pad_idx) | (x == vocab.eos_idx) | (x == vocab.sos_idx)
        
        # find not special tokens in target
        tgt_not_pad = torch.where(is_special_F(tgt), False, True)
        src_not_pad = torch.where(is_special_F(src), False, True)
        src_lens = src_not_pad.sum(1)
        tgt_lens = tgt_not_pad.sum(1)
        
        # provide target sequence length
        if length_oracle:
            predictions = predict_batch_oracle(model, src, vocab, tgt_lens, device=device)
        # no target sequence length
        else:
            predictions = predict_batch(model, src, vocab, device=device)
        # Remove SOS token for comparison
        predictions = predictions[:,1:]
        # pad generated to match tgt length
        predictions = torch.nn.functional.pad(predictions,
                                                (0, tgt.size(1) - predictions.size(1)),
                                                value=vocab.pad_idx)
        
        # exact sequence match: matching seqs/total seqs
        # disregarding special tokens
        # find tokenwise differences
        tok_diff = torch.where(predictions != tgt , True, False)
        # special tokens 
        tgt_pad = torch.where(is_special_F(tgt), True, False)
        # remove special tokens
        tok_diff = torch.logical_and(tok_diff, ~tgt_pad)
        # calc # differences pr sequence
        seq_diff = tok_diff.sum(dim=1)
        # sum up seq differences, if no differences then correct
        correct_seqs = torch.where(seq_diff == 0, 1, 0)

        correct[Flag.Avrg][Flag.SL] += correct_seqs.sum()
        total[Flag.Avrg][Flag.SL] += tgt.size(0)

        
        correct[Flag.InLen][Flag.SL].scatter_add_(0, src_lens ,  correct_seqs)
        total[Flag.InLen][Flag.SL].scatter_add_(0, src_lens ,  torch.ones_like(src_lens))#torch.scatter_add(total[MetricFlag.SLIL],0, src_lens ,  torch.ones_like(src_lens))

        correct[Flag.OutLen][Flag.SL].scatter_add_(0, tgt_lens ,  correct_seqs)
        total[Flag.OutLen][Flag.SL].scatter_add_(0, tgt_lens ,  torch.ones_like(tgt_lens))


        # token level accuracy: matching tokens/total tokens
        # disregarding special tokens
        # numb of target tokens to predict
        tokens_to_predict = tgt_lens.sum()
        # find not special tokens in both target and predictions
        both_tokens = torch.where(is_special_F(tgt) & is_special_F(predictions), False, True)
        # correctly predicted not special tokens
        correctly_predicted = torch.where(both_tokens & (tgt == predictions), True, False)
        correct[Flag.Avrg][Flag.TL] += correctly_predicted.sum()
        total[Flag.Avrg][Flag.TL] += tokens_to_predict

        correct[Flag.InLen][Flag.TL].scatter_add_(0, src_lens ,  correctly_predicted.sum(1))
        total[Flag.InLen][Flag.TL].scatter_add_(0, src_lens ,  tgt_lens)

        correct[Flag.OutLen][Flag.TL].scatter_add_(0, tgt_lens ,  correctly_predicted.sum(1))
        total[Flag.OutLen][Flag.TL].scatter_add_(0, tgt_lens ,  tgt_lens)


    
    err_rate = calc_error_rate(correct, total)
    accuracy = calc_acc(correct, total)

    result = {
        Flags.MetricFlags.CORRECT : detach_result_map(correct),
        Flags.MetricFlags.TOTAL : detach_result_map(total),
        Flags.MetricFlags.ACC : detach_result_map(accuracy),
        Flags.MetricFlags.ERR : detach_result_map(err_rate)
    }
    return result

def evaluate_model_batchwise(model, test_data, test_loader, vocab, device="cpu"): 
    no_oracle = _evaluate_model_batchwise(model,
                                            test_data,
                                            test_loader,
                                            vocab,
                                            length_oracle=False,
                                            device=device,
                                            )
    
    oracle = _evaluate_model_batchwise(model,
                                        test_data, 
                                        test_loader, 
                                        vocab, 
                                        length_oracle=True,
                                        device=device,
                                        )
    result = {Flags.PredictionFlags.ORACLE: oracle,
                Flags.PredictionFlags.NO_ORACLE: no_oracle}
    
    return result

def calc_error_rate(correct, total):
    err_rate = {}
    # total - correct / total
    err_rate[Flag.Avrg][Flag.TL] = ((total[Flag.Avrg][Flag.TL]- correct[Flag.Avrg][Flag.TL])/ total[Flag.Avrg][Flag.TL])
    err_rate[Flag.Avrg][Flag.SL] = ((total[Flag.Avrg][Flag.SL]- correct[Flag.Avrg][Flag.SL]) / total[Flag.Avrg][Flag.SL])
    err_rate[Flag.InLen][Flag.TL] = torch.div((total[Flag.InLen][Flag.TL] - correct[Flag.InLen][Flag.TL]), total[Flag.InLen][Flag.TL])
    err_rate[Flag.InLen][Flag.SL] = torch.div((total[Flag.InLen][Flag.SL] - correct[Flag.InLen][Flag.SL]), total[Flag.InLen][Flag.SL])
    err_rate[Flag.OutLen][Flag.TL] = torch.div((total[Flag.OutLen][Flag.TL] - correct[Flag.OutLen][Flag.TL]), total[Flag.OutLen][Flag.TL])
    err_rate[Flag.OutLen][Flag.SL] = torch.div((total[Flag.Avrg][Flag.SL] - correct[Flag.OutLen][Flag.SL]), total[Flag.Avrg][Flag.SL])
    return err_rate

def calc_acc(correct, total):
    acc = {}
    # correct / total
    acc[Flag.Avrg][Flag.TL] = (correct[Flag.Avrg][Flag.TL]/ total[Flag.Avrg][Flag.TL])
    acc[Flag.Avrg][Flag.SL] = (correct[Flag.Avrg][Flag.SL] / total[Flag.Avrg][Flag.SL])
    acc[Flag.InLen][Flag.TL] = torch.div(correct[Flag.InLen][Flag.TL], total[Flag.InLen][Flag.TL])
    acc[Flag.InLen][Flag.SL] = torch.div(correct[Flag.InLen][Flag.SL], total[Flag.InLen][Flag.SL])
    acc[Flag.OutLen][Flag.TL] = torch.div(correct[Flag.OutLen][Flag.TL], total[Flag.OutLen][Flag.TL])
    acc[Flag.OutLen][Flag.SL] = torch.div(correct[Flag.OutLen][Flag.SL], total[Flag.Avrg][Flag.SL])
    return acc

def detach_result_map(m):
    if isinstance(m[Flag.Avrg][Flag.TL], torch.Tensor):
        m[Flag.Avrg][Flag.TL] = m[Flag.Avrg][Flag.TL].item()
    if isinstance(m[Flag.Avrg][Flag.SL], torch.Tensor):
        m[Flag.Avrg][Flag.SL] = m[Flag.Avrg][Flag.SL].item()
    m[Flag.InLen][Flag.TL] = remove_nan(m[Flag.InLen][Flag.TL]).cpu().numpy()
    m[Flag.InLen][Flag.SL] = remove_nan(m[Flag.InLen][Flag.SL]).cpu().numpy()
    m[Flag.OutLen][Flag.TL] = remove_nan(m[Flag.OutLen][Flag.TL]).cpu().numpy()
    m[Flag.OutLen][Flag.SL] = remove_nan(m[Flag.OutLen][Flag.SL]).cpu().numpy()
    return m

def remove_nan(t):
    t = torch.where(torch.isnan(t), 0, t)
    return t


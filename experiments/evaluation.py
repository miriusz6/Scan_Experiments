import torch
from experiments.metric import MetricFlag, MetricType, OracleFlag
from copy import deepcopy

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

def _evaluate_model_batchwise(model, test_data, test_loader, vocab, device="cpu", length_oracle=False):
    inp_len_max = test_data.command_length_span[1]
    tgt_len_max = test_data.action_sequence_span[1]
    inp_seq_span = torch.zeros(inp_len_max+1, device=device, dtype=torch.long)
    tgt_seq_span = torch.zeros(tgt_len_max+1, device=device, dtype=torch.long)

    template = {
        MetricFlag.TL: 0,
        MetricFlag.SL: 0,
        MetricFlag.TLIL: inp_seq_span.clone(),
        MetricFlag.SLIL: inp_seq_span.clone(),
        MetricFlag.TLOL: tgt_seq_span.clone(),
        MetricFlag.SLOL: tgt_seq_span.clone(),
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

        correct[MetricFlag.SL] += correct_seqs.sum()
        total[MetricFlag.SL] += tgt.size(0)

        
        correct[MetricFlag.SLIL].scatter_add_(0, src_lens ,  correct_seqs)
        total[MetricFlag.SLIL].scatter_add_(0, src_lens ,  torch.ones_like(src_lens))#torch.scatter_add(total[MetricFlag.SLIL],0, src_lens ,  torch.ones_like(src_lens))

        correct[MetricFlag.SLOL].scatter_add_(0, tgt_lens ,  correct_seqs)
        total[MetricFlag.SLOL].scatter_add_(0, tgt_lens ,  torch.ones_like(tgt_lens))


        # token level accuracy: matching tokens/total tokens
        # disregarding special tokens
        # numb of target tokens to predict
        tokens_to_predict = tgt_lens.sum()
        # find not special tokens in both target and predictions
        both_tokens = torch.where(is_special_F(tgt) & is_special_F(predictions), False, True)
        # correctly predicted not special tokens
        correctly_predicted = torch.where(both_tokens & (tgt == predictions), True, False)
        correct[MetricFlag.TL] += correctly_predicted.sum()
        total[MetricFlag.TL] += tokens_to_predict

        correct[MetricFlag.TLIL].scatter_add_(0, src_lens ,  correctly_predicted.sum(1))
        total[MetricFlag.TLIL].scatter_add_(0, src_lens ,  tgt_lens)

        correct[MetricFlag.TLOL].scatter_add_(0, tgt_lens ,  correctly_predicted.sum(1))
        total[MetricFlag.TLOL].scatter_add_(0, tgt_lens ,  tgt_lens)


    
    err_rate = calc_error_rate(correct, total)
    accuracy = calc_acc(correct, total)

    result = {
        MetricType.CORRECT : detach_result_map(correct),
        MetricType.TOTAL : detach_result_map(total),
        MetricType.ACC : detach_result_map(accuracy),
        MetricType.ERR : detach_result_map(err_rate)
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
    result = {OracleFlag.ORACLE: oracle,
                OracleFlag.NO_ORACLE: no_oracle}
    
    return result

def calc_error_rate(correct, total):
    err_rate = {}
    err_rate[MetricFlag.TL] = ((total[MetricFlag.TL]- correct[MetricFlag.TL])/ total[MetricFlag.TL])
    err_rate[MetricFlag.SL] = ((total[MetricFlag.SL]-correct[MetricFlag.SL]) / total[MetricFlag.SL])
    err_rate[MetricFlag.TLIL] = torch.div((total[MetricFlag.TLIL] - correct[MetricFlag.TLIL]), total[MetricFlag.TLIL])
    err_rate[MetricFlag.SLIL] = torch.div((total[MetricFlag.SLIL] - correct[MetricFlag.SLIL]), total[MetricFlag.SLIL])
    err_rate[MetricFlag.TLOL] = torch.div((total[MetricFlag.TLOL] - correct[MetricFlag.TLOL]), total[MetricFlag.TLOL])
    err_rate[MetricFlag.SLOL] = torch.div((total[MetricFlag.SL] - correct[MetricFlag.SLOL]), total[MetricFlag.SL])
    return err_rate

def calc_acc(correct, total):
    acc = {}
    acc[MetricFlag.TL] = (correct[MetricFlag.TL]/ total[MetricFlag.TL])
    acc[MetricFlag.SL] = (correct[MetricFlag.SL] / total[MetricFlag.SL])
    acc[MetricFlag.TLIL] = torch.div(correct[MetricFlag.TLIL], total[MetricFlag.TLIL])
    acc[MetricFlag.SLIL] = torch.div(correct[MetricFlag.SLIL], total[MetricFlag.SLIL])
    acc[MetricFlag.TLOL] = torch.div(correct[MetricFlag.TLOL], total[MetricFlag.TLOL])
    acc[MetricFlag.SLOL] = torch.div(correct[MetricFlag.SLOL], total[MetricFlag.SL])
    return acc

def detach_result_map(m):
    if isinstance(m[MetricFlag.TL], torch.Tensor):
        m[MetricFlag.TL] = m[MetricFlag.TL].item()
    if isinstance(m[MetricFlag.SL], torch.Tensor):
        m[MetricFlag.SL] = m[MetricFlag.SL].item()
    m[MetricFlag.TLIL] = remove_nan(m[MetricFlag.TLIL]).cpu().numpy()
    m[MetricFlag.SLIL] = remove_nan(m[MetricFlag.SLIL]).cpu().numpy()
    m[MetricFlag.TLOL] = remove_nan(m[MetricFlag.TLOL]).cpu().numpy()
    m[MetricFlag.SLOL] = remove_nan(m[MetricFlag.SLOL]).cpu().numpy()
    return m

def remove_nan(t):
    t = torch.where(torch.isnan(t), 0, t)
    return t


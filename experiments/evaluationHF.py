import re
import torch
from experiments.metric import Flags
from torch.nn.utils import clip_grad_norm_

from transformers import PreTrainedTokenizer

def predict_batch(model, src_sequences, src_msks, tokenizer:PreTrainedTokenizer,
                   max_len=148, device="cpu"):
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

    eos = tokenizer.convert_tokens_to_ids(['<s\>'])
    bos = tokenizer.convert_tokens_to_ids(['<s>'])

    model.eval()
    with torch.no_grad():
        # Initialize target sequence with SOS token
        #tgt = [vocab.sos_idx] + [vocab.pad_idx] * (max_len)
        tgt = [bos] + ([tokenizer.pad_token_type_id] * (max_len))
        tgt = torch.tensor([tgt], device=device)
        tgt = tgt.repeat(src_sequences.size(0), 1)

        tgt_msks = ([1] + [tokenizer.pad_token_type_id])*max_len
        tgt_msks = tgt_msks.repeat(src_sequences.size(0), 1)

        # holds indx of sequences containing EOS token
        finished = torch.tensor([False]*src_sequences.size(0), device=device)
        
        # Generate tokens one by one
        for i in range(1,max_len):
            tgt_msks[:,i] = 1
            # indx of batch dim where EOS token has not been generated
            active_indxs = torch.where(finished == False)[0]
            # feed only unfinished sequences to decoder
            # remove padding
            tgt_active = tgt_active[:,0:i]
            m = {
                "input_ids": src_sequences[active_indxs],
                "attention_mask" : src_msks[active_indxs],
                "decoder_input_ids" : tgt[active_indxs],
                "decoder_attention_mask" : tgt_msks[active_indxs],
                "head_mask" : None,
                "decoder_head_mask" : None,
                "cross_attn_head_mask" : None,
                "encoder_outputs" : None,
                "past_key_values" : None,
                "inputs_embeds" : None,
                "decoder_inputs_embeds" : None,
                "use_cache" : None,
                "output_attentions" : None,
                "output_hidden_states" : None,
                "return_dict" : None,
            }
            out = model(m)

            # Get next token prediction
            next_token = out.argmax(dim=-1)
            next_token = next_token[:, -1]
            # store new prediction for active sequences
            tgt[active_indxs,i] = next_token
            
            # update finished sequences if any EOS token is generated
            new_finished = torch.where(next_token == eos, True, False)
            finished[active_indxs] = torch.logical_or(finished[active_indxs], new_finished)

            # early stopping if all sequences produced EOS token
            if finished.all():
                break
    return tgt

# def predict_batch_oracle(model, src_sequence, vocab, seq_lenghts, device="cpu"):
#     """
#     Predicts a batch of sequences using the given model and source sequences.
#     Args:
#         model (torch.nn.Module): The model to use for prediction.
#         src_sequence (torch.Tensor): The source sequences to predict from.
#         vocab (Vocabulary): The vocabulary object containing all tokens.
#         seq_lenghts (list[int]): The lengths of the sequences to predict.
#         device (str, optional): The device to run the prediction on. Defaults to "cpu".
#     Returns:
#         torch.Tensor: The predicted target sequences.
#     """

#     model.eval()
#     with torch.no_grad():
#         # Prepare source sequence
#         src = src_sequence.to(device) 
#         src_mask = model.create_src_mask(src)

#         # Generate initial encoder output
#         encoder_output = model.encoder(src, src_mask)
#         #seq_lenghts += 1
#         longest = max(seq_lenghts)
#         # Initialize target sequence with SOS token
#         tgt = [vocab.sos_idx] + [vocab.pad_idx] * (longest)
#         tgt = torch.tensor([tgt], device=device)
#         tgt = tgt.repeat(src.size(0), 1)

#         # holds indx of sequences containing with target length
#         finished = torch.tensor([False]*src.size(0), device=device)
        
#         # Generate tokens one by one
#         for i in range(1,longest):
#             # indx of batch dim where seq len is not yet matched
#             active_indxs = torch.where(finished == False)[0]
#             # feed only unfinished sequences to decoder
#             tgt_active = tgt[active_indxs]
#             # remove padding
#             tgt_active = tgt_active[:,0:i]
#             # create mask
#             tgt_mask = model.create_tgt_mask(tgt_active)
#             # decode
#             decoder_output = model.decoder(tgt_active,
#                                            encoder_output[active_indxs],
#                                            src_mask[active_indxs],
#                                            tgt_mask)
#             # consider only prediction for last token
#             decoder_output = decoder_output[:,-1,:]
#             # Get next token prediction
#             next_token = decoder_output.argmax(dim=-1)
#             # seqences with target length
#             new_finished = torch.where(seq_lenghts[active_indxs] == (i), True, False)
#             # if EOS token is predicted, use second best prediction
#             early_eos = torch.where((next_token == vocab.eos_idx),True, False)
#             if early_eos.any():
#                 early_eos_indx = next_token[early_eos]
#                 decoder_output[:,early_eos_indx] = 0
#                 next_token = decoder_output.argmax(dim=-1)

#             # store new prediction for active sequences
#             tgt[active_indxs,i] = next_token
#             # update finished sequences 
#             finished[active_indxs] = torch.logical_or(finished[active_indxs], new_finished)

#             # early stopping if all sequences produced EOS token
#             if finished.all():
#                 break
#     return tgt

  
# def _evaluate_model_batchwise(model, test_data, test_loader, vocab, device="cpu", length_oracle=False):
#     inp_len_max = test_data.command_length_span[1]
#     tgt_len_max = test_data.action_sequence_span[1]
#     inp_seq_span = torch.zeros(inp_len_max+1, device=device, dtype=torch.long)
#     tgt_seq_span = torch.zeros(tgt_len_max+1, device=device, dtype=torch.long)
#     zero_item = torch.tensor(0, device=device)
#     # TO: Total, CR: Correct
#     # TL: Token Level, SL: Sequence Level
#     # IL: InLen, OL: OutLen, SU: Sum
#     TO_TL_IL, CR_TL_IL = inp_seq_span.clone(), inp_seq_span.clone()
#     TO_SL_IL, CR_SL_IL = inp_seq_span.clone(), inp_seq_span.clone()
#     TO_TL_OL, CR_TL_OL = tgt_seq_span.clone(), tgt_seq_span.clone()
#     TO_SL_OL, CR_SL_OL = tgt_seq_span.clone(), tgt_seq_span.clone()
#     TO_TL_SU, CR_TL_SU = zero_item.clone(), zero_item.clone()
#     TO_SL_SU, CR_SL_SU = zero_item.clone(), zero_item.clone()

#     for src, _, tgt in test_loader:
#         src = src.to(device)
#         is_special_F = lambda x: (x == vocab.pad_idx) | (x == vocab.eos_idx) | (x == vocab.sos_idx)
        
#         # find not special tokens in target
#         tgt_not_pad = torch.where(is_special_F(tgt), False, True)
#         src_not_pad = torch.where(is_special_F(src), False, True)
#         src_lens = src_not_pad.sum(1)
#         tgt_lens = tgt_not_pad.sum(1)
        
#         # provide target sequence length
#         if length_oracle:
#             predictions = predict_batch_oracle(model, src, vocab, tgt_lens, device=device)
#         # no target sequence length
#         else:
#             predictions = predict_batch(model, src, vocab, device=device)
#         # Remove SOS token for comparison
#         predictions = predictions[:,1:]
#         # pad generated to match tgt length
#         predictions = torch.nn.functional.pad(predictions,
#                                                 (0, tgt.size(1) - predictions.size(1)),
#                                                 value=vocab.pad_idx)
        
#         # exact sequence match: matching seqs/total seqs
#         # disregarding special tokens
#         # find tokenwise differences
#         tok_diff = torch.where(predictions != tgt , True, False)
#         # special tokens 
#         tgt_pad = torch.where(is_special_F(tgt), True, False)
#         # remove special tokens
#         tok_diff = torch.logical_and(tok_diff, ~tgt_pad)
#         # calc # differences pr sequence
#         seq_diff = tok_diff.sum(dim=1)
#         # sum up seq differences, if no differences then correct
#         correct_seqs = torch.where(seq_diff == 0, 1, 0)

#         CR_SL_SU += correct_seqs.sum()
#         TO_SL_SU += tgt.size(0)
        
#         CR_SL_IL.scatter_add_(0, src_lens, correct_seqs)
#         TO_SL_IL.scatter_add_(0, src_lens, torch.ones_like(src_lens))

#         CR_SL_OL.scatter_add_(0, tgt_lens, correct_seqs)
#         TO_SL_OL.scatter_add_(0, tgt_lens, torch.ones_like(tgt_lens))

#         # token level accuracy: matching tokens/total tokens
#         # disregarding special tokens
#         # numb of target tokens to predict
#         tokens_to_predict = tgt_lens.sum()
#         # find not special tokens in both target and predictions
#         both_tokens = torch.where(is_special_F(tgt) & is_special_F(predictions), False, True)
#         # correctly predicted not special tokens
#         correctly_predicted = torch.where(both_tokens & (tgt == predictions), True, False)
#         CR_TL_SU += correctly_predicted.sum()
#         TO_TL_SU += tokens_to_predict

#         CR_TL_IL.scatter_add_(0, src_lens ,  correctly_predicted.sum(1))
#         TO_TL_IL.scatter_add_(0, src_lens ,  tgt_lens)

#         CR_TL_OL.scatter_add_(0, tgt_lens ,  correctly_predicted.sum(1))
#         TO_TL_OL.scatter_add_(0, tgt_lens ,  tgt_lens)

#     # Create result dictionary
#     err_rate = {
#     Flags.DistrFlags.InLen:
#         {
#             Flags.LevelFlags.TL: calc_error_rate_scalar(CR_TL_IL, TO_TL_IL).cpu().numpy(),
#             Flags.LevelFlags.SL: calc_error_rate_scalar(CR_SL_IL, TO_SL_IL).cpu().numpy(),
#         },
#     Flags.DistrFlags.OutLen:
#         {
#             Flags.LevelFlags.TL: calc_error_rate_tensor(CR_TL_OL, TO_TL_OL).cpu().numpy(),
#             Flags.LevelFlags.SL: calc_error_rate_tensor(CR_SL_OL, TO_SL_OL).cpu().numpy(),
#         },
#     Flags.DistrFlags.Avrg:
#         {
#             Flags.LevelFlags.TL: calc_error_rate_scalar(CR_TL_SU, TO_TL_SU).item(),
#             Flags.LevelFlags.SL: calc_error_rate_scalar(CR_SL_SU, TO_SL_SU).item(),
#         }
#     }

#     accuracy = {
#     Flags.DistrFlags.InLen:
#         {
#             Flags.LevelFlags.TL: calc_acc_scalar(CR_TL_IL, TO_TL_IL).cpu().numpy(),
#             Flags.LevelFlags.SL: calc_acc_scalar(CR_SL_IL, TO_SL_IL).cpu().numpy(),
#         },
#     Flags.DistrFlags.OutLen:
#         {
#             Flags.LevelFlags.TL: calc_acc_tensor(CR_TL_OL, TO_TL_OL).cpu().numpy(),
#             Flags.LevelFlags.SL: calc_acc_tensor(CR_SL_OL, TO_SL_OL).cpu().numpy(),
#         },
#     Flags.DistrFlags.Avrg:
#         {
#             Flags.LevelFlags.TL: calc_acc_scalar(CR_TL_SU, TO_TL_SU).item(),
#             Flags.LevelFlags.SL: calc_acc_scalar(CR_SL_SU, TO_SL_SU).item(),
#         }
#     }

#     correct = {
#     Flags.DistrFlags.InLen:
#         {
#             Flags.LevelFlags.TL: CR_TL_IL.cpu().numpy(),
#             Flags.LevelFlags.SL: CR_SL_IL.cpu().numpy(),
#         },
#     Flags.DistrFlags.OutLen:
#         {
#             Flags.LevelFlags.TL: CR_TL_OL.cpu().numpy(),
#             Flags.LevelFlags.SL: CR_SL_OL.cpu().numpy(),
#         },
#     Flags.DistrFlags.Sum:
#         {
#             Flags.LevelFlags.TL: CR_TL_SU.item(),
#             Flags.LevelFlags.SL: CR_SL_SU.item(),
#         }
#     }

#     total = {
#     Flags.DistrFlags.InLen:
#         {
#             Flags.LevelFlags.TL: TO_TL_IL.cpu().numpy(),
#             Flags.LevelFlags.SL: TO_SL_IL.cpu().numpy(),
#         },
#     Flags.DistrFlags.OutLen:
#         {
#             Flags.LevelFlags.TL: TO_TL_OL.cpu().numpy(),
#             Flags.LevelFlags.SL: TO_SL_OL.cpu().numpy(),
#         },
#     Flags.DistrFlags.Sum:
#         {
#             Flags.LevelFlags.TL: TO_TL_SU.item(),
#             Flags.LevelFlags.SL: TO_SL_SU.item(),
#         }
#     }

#     result = {
#         Flags.MetricFlags.CORRECT : correct,
#         Flags.MetricFlags.TOTAL : total,
#         Flags.MetricFlags.ACC : accuracy,
#         Flags.MetricFlags.ERR : err_rate,
#     }
#     return result

# def evaluate_model_batchwise(model, test_data, test_loader, vocab, device="cpu"): 
#     no_oracle = _evaluate_model_batchwise(model,
#                                             test_data,
#                                             test_loader,
#                                             vocab,
#                                             length_oracle=False,
#                                             device=device,
#                                             )
    
#     # oracle = _evaluate_model_batchwise(model,
#     #                                     test_data, 
#     #                                     test_loader, 
#     #                                     vocab, 
#     #                                     length_oracle=True,
#     #                                     device=device,
#     #                                     )
#     result = {Flags.PredictionFlags.ORACLE: None, #oracle,
#                 Flags.PredictionFlags.NO_ORACLE:  no_oracle}
    
#     return result

# def calc_error_rate_scalar(correct, total):
#     err_rate = (total - correct) / total
#     return err_rate

# def calc_error_rate_tensor(correct, total):
#     err_rate = remove_nan(torch.div((total - correct), total))
#     return err_rate

# def calc_acc_scalar(correct, total):
#     acc = correct / total
#     return acc

# def calc_acc_tensor(correct, total):
#     acc = remove_nan(torch.div(correct, total))
#     return acc

# def remove_nan(t):
#     t = torch.where(torch.isnan(t), 0, t)
#     return t


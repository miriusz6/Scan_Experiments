import matplotlib.pyplot as plt
import numpy as np
import re
from experiments.evaluation_result import EvaluationResult
from experiments.evaluation_result_container import EvaluationResultContainer
from experiments.metric import MetricTemplate, Flags
from experiments.evaluation_statistics import  aggregate_by_subname
from experiments.evaluation_statistics import aggregate_by_subname
from math import log10, floor

# def prepare_scalars_for_plot(
#                       results:EvaluationResultContainer|EvaluationResult, 
#                       template:MetricTemplate, 
#                       xs = None,
#                       x_lables = None,
#                       ):
#     data = results.get_data(template)
#     if not x_lables and not x_lbls_re_patt:
#         x_lables = np.arange(len(data))
#     elif x_lbls_re_patt:
#         x_lables = [re.search(x_lbls_re_patt, m.name).group(0) for m in data]
#     if not xs :
#         xs = np.arange(len(data))
#     ys = [m.val for m in data]
#     xtics = np.arange(len(data))
#     return xs, ys, x_lables, xtics

from experiments.metric import Metric

def prepare_for_plot(
                      data: list[Metric]|Metric,
                      xs = None,
                      x_lables = None,
                      cut_trailing_zeros = False
                      ):
    
    is_lst = isinstance(data,list)
    
    if is_lst:
        arr_data = isinstance(data[0].val, np.ndarray)
        if arr_data and len(data) > 1:
                raise ValueError("Only one array metric is allowed")
        elif arr_data:
            data = data[0].val
        else:
            data = [m.val for m in data]
    else:
        data = data.val
        
    
            
    
    
    if x_lables is None:
        x_lables = np.arange(len(data))
    if not xs:
        xs = np.arange(len(data))
    ys = data
    # if arr_data:
    #     ys = data
    # else:
    #     ys = [m.val for m in data]
    xtics = np.arange(len(data))
    
    head_idx = 0
    tail_idx = len(data)
    
    if not cut_trailing_zeros:
        return xs, ys, x_lables, xtics
    
    head_idx = -1
    tail_idx = -1
    for i,v in enumerate(data):
        if head_idx < 0 and v > 0:
            head_idx = i
            break
    #iiterate in reverse
    for i,v in enumerate(data[::-1]):
        if tail_idx < 0 and v > 0:
            tail_idx = len(data) - i
            break
        
    xtics = xtics[head_idx:tail_idx]
    xs = xs[head_idx:tail_idx]
    ys = ys[head_idx:tail_idx]
    x_lables = x_lables[head_idx:tail_idx]
        
                
    
    return xs, ys, x_lables, xtics






def plot_k_fold_reps(ax,
                    rep1:EvaluationResultContainer,
                    rep2:EvaluationResultContainer,
                    rep3:EvaluationResultContainer,
                    template:MetricTemplate,
                    x_lbls
                     ):

    # averages over three folds
    rep1_fld_avrg = aggregate_by_subname([rep1], id_sub_name="fold", id_range=(1, 3), reps=1)
    rep2_fld_avrg = aggregate_by_subname([rep2], id_sub_name="fold", id_range=(1, 3), reps=1)
    rep3_fld_avrg = aggregate_by_subname([rep3], id_sub_name="fold", id_range=(1, 3), reps=1)
    # averages over the three repetitions of the 3-fold cross validation
    rep_avrg_fld_avrg = aggregate_by_subname([rep1,rep2,rep3], id_sub_name="fold", id_range=(1, 3), reps=3)

    # prepare scalars

    r1_ms:list[Metric] = rep1_fld_avrg.get_data(template)
    xs, ys, x_lables, xtics = prepare_for_plot(r1_ms, x_lables=x_lbls)
    ax.plot(xs,ys,  linestyle='-', color='blue', label="repetition 1", alpha = 0.5)

    r2_ms:list[Metric] = rep2_fld_avrg.get_data(template)
    xs, ys, x_lables, xtics = prepare_for_plot(r2_ms, x_lables=x_lbls)
    ax.plot(xs,ys,  linestyle='-', color='green', label="repetition 2", alpha = 0.5)

    r3_ms:list[Metric] = rep3_fld_avrg.get_data(template)
    xs, ys, x_lables, xtics = prepare_for_plot(r3_ms, x_lables=x_lbls)
    ax.plot(xs,ys,  linestyle='-', color='brown', label="repetition 3", alpha = 0.5)

    avrg_ms:list[Metric] = rep_avrg_fld_avrg.get_data(template)
    xs, ys, x_lables, xtics = prepare_for_plot(avrg_ms, x_lables=x_lbls)
    ax.plot(xs,ys, marker='o', linestyle='-', color='orange', label="average")

    # Customize plot
    ax.set_xticks(xtics, x_lables)
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Error (%)")
    ax.legend()
    return rep_avrg_fld_avrg

err_or_tok = [
Flags.PredictionFlags.ORACLE,
Flags.LevelFlags.TL,
Flags.DistrFlags.Avrg,
Flags.MetricFlags.ERR,
]

err_tok = [
    Flags.PredictionFlags.NO_ORACLE,
    Flags.LevelFlags.TL,
    Flags.DistrFlags.Avrg,
    Flags.MetricFlags.ERR,
]

err_or_seq = [
    Flags.PredictionFlags.ORACLE,
    Flags.LevelFlags.SL,
    Flags.DistrFlags.Avrg,
    Flags.MetricFlags.ERR,
]

err_seq = [
    Flags.PredictionFlags.NO_ORACLE,
    Flags.LevelFlags.SL,
    Flags.DistrFlags.Avrg,
    Flags.MetricFlags.ERR,
]

err_or_tok_temp = MetricTemplate(flags=err_or_tok)
err_tok_temp = MetricTemplate(flags=err_tok)
err_or_seq_temp = MetricTemplate(flags=err_or_seq)
err_seq_temp = MetricTemplate(flags=err_seq)


def mk_3fold_3reps_fig(f3_r1, f3_r2, f3_r3, epoch_range:tuple[int,int,int]):
    fig, axs = plt.subplots(5, 1, figsize=(15,35))
    ax_err_tok = axs[0]
    ax_err_or_tok = axs[1]
    ax_err_seq = axs[2]
    ax_err_or_seq = axs[3]
    ax_err_by_flags = axs[4]

    x_lbls = range(epoch_range[0], epoch_range[1], epoch_range[2])
    
    #ax_err_tok.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_tok, f3_r1, f3_r2, f3_r3,template=err_tok_temp, x_lbls=x_lbls)
    ax_err_tok.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_tok.__str__(),
                            fontsize = 10)

    #ax_err_or_tok.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_or_tok, f3_r1, f3_r2, f3_r3,template=err_or_tok_temp, x_lbls=x_lbls)
    ax_err_or_tok.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_or_tok.__str__(),
                            fontsize = 10)

    #ax_err_seq.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_seq, f3_r1, f3_r2, f3_r3,template=err_seq_temp, x_lbls=x_lbls)
    ax_err_seq.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_seq.__str__(),
                            fontsize = 10)

    #ax_err_or_seq.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_or_seq, f3_r1, f3_r2, f3_r3,template=err_or_seq_temp, x_lbls=x_lbls)
    ax_err_or_seq.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_or_seq.__str__(),
                            fontsize = 10)


    
    
    rep_avrg_fld_avrg = aggregate_by_subname([f3_r1, f3_r2, f3_r3], id_sub_name="fold", id_range=(1, 3), reps=3)

    err_tok_ms:list[Metric] = rep_avrg_fld_avrg.get_data(err_tok_temp)
    xs, ys, x_lables, xtics = prepare_for_plot(err_tok_ms, x_lables=x_lbls)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='green', label=str(err_tok))

    err_or_tok_ms:list[Metric] = rep_avrg_fld_avrg.get_data(err_or_tok_temp)
    xs, ys, x_lables, xtics = prepare_for_plot(err_or_tok_ms, x_lables=x_lbls)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='red', label=str(err_or_tok))

    err_seq_ms:list[Metric] = rep_avrg_fld_avrg.get_data(err_seq_temp)
    xs, ys, x_lables, xtics = prepare_for_plot(err_seq_ms, x_lables=x_lbls)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='orange', label=str(err_seq))

    err_or_seq_ms:list[Metric] = rep_avrg_fld_avrg.get_data(err_or_seq_temp)
    xs, ys, x_lables, xtics = prepare_for_plot(err_or_seq_ms, x_lables=x_lbls)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='purple', label=str(err_or_seq))

    ax_err_by_flags.set_xticks(xtics,x_lables)

    ax_err_by_flags.set_xlabel("Epoch number")
    ax_err_by_flags.set_ylabel("Error (%)")
    ax_err_by_flags.legend()

    return fig, axs



def mk_bar_plot(ax:plt.axes, 
                data: list[Metric]|Metric,
                x_lbls= None,
                coords_txt = True,
                h_lines = True,
                cut_trailing_zeros = True,
                omit_zero_lbls = False,
                add_avrg_line = True,
                bar_kwrgs = {}
                ):
                
    xs, ys, x_lables, xtics = prepare_for_plot(
                                data=data,
                                x_lables=x_lbls,
                                cut_trailing_zeros=cut_trailing_zeros
                                )
    to_del = []
    if omit_zero_lbls:
        for i,y in enumerate(ys):
            if y == 0:
                to_del.append(i)
    
    x_lables = np.delete(x_lables, to_del)
    xtics = np.delete(xtics, to_del)
        
    x_max = max(xs)+1
    x_min = min(xs)-1
    
    bar_act = ax.bar(xs, ys, **bar_kwrgs)
    hlins_act = None
    coors_txt_act = None
    avrg_line_act = None

    if h_lines:
        h_ls = calc_h_lines(ys)
        hlins_act = ax.hlines(h_ls,xmin=x_min, xmax=x_max, colors = "grey", alpha = 0.4)
        ax.set_yticks(h_ls)
        ax.set_yticklabels(h_ls)
    if coords_txt:
        for i in range(len(ys)):
            if ys[i] > 0:
                coors_txt_act = ax.text(xs[i], ys[i], f"{ys[i]:.2f}", ha='center', va='bottom')
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(xtics)
    ax.set_xticklabels(x_lables)
    
    if add_avrg_line:
        buff = [ys[i] for i in range(len(ys)) if i not in to_del]
        avrg = np.mean(buff)
        avrg_line_act = ax.hlines(avrg,xmin=x_min, xmax=x_max, colors = "red", alpha = 0.7, label = "Avrg")
        ax.text(x_max, avrg, f"{avrg:.2f}", ha='center', va='bottom',fontdict = {'color':'red'})
    
    return (bar_act, hlins_act, coors_txt_act, avrg_line_act)
    
def mk_bar_plt_omit_empty(ax:plt.axes, 
                data_metric:Metric,
                total_metric: Metric,
                coords_txt = True,
                h_lines = True,
                cut_trailing_zeros = True,
                omit_zero_lbls = False,
                add_avrg_line = True,
                bar_kwrgs = {}
                ):
    
    total_arr = total_metric.val
    non_zero_indx = [i for i,v in enumerate(total_arr) if v > 0]
    v_arr = data_metric.val
    v_arr = v_arr[non_zero_indx]
    m = Metric(v_arr,'',None,data_metric.flags)
    actors = mk_bar_plot(ax, m,
                        x_lbls=non_zero_indx,
                        coords_txt=coords_txt,
                        h_lines=h_lines,
                        bar_kwrgs=bar_kwrgs,
                        cut_trailing_zeros=cut_trailing_zeros,
                        omit_zero_lbls=omit_zero_lbls,
                        add_avrg_line=add_avrg_line
                        )   
    return actors
    


    
def calc_h_lines(data):
    l = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9,10]
    l = np.array(l)
    if max(data) > 1:
        bfr_dot = str(max(data)).split(".")[0]
        tens = len(bfr_dot)-1
        l = l * (10**(tens)) 
        # to int
        l = [i//1 for i in l]
    else:
        aftr_dot = str(max(data)).split(".")[1]
        zeros = 0
        for c in aftr_dot:
            if c == "0":
                zeros += 1
            else:
                break
        tens = zeros+1
        l = l * (10**(-tens)) 
        l = [round(i, tens) for i in l]
    return l
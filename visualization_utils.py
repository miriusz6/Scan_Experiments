import matplotlib.pyplot as plt
import numpy as np
from experiments.evaluation_result_container import EvaluationResultContainer
from experiments.metric import MetricTemplate, Flags, _MetricFlags
from experiments.evaluation_statistics import  aggregate_by_subname
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


def _mk_bar_plot(ax:plt.axes, 
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
    

def mk_bar_plt(ax:plt.axes, 
                data_metric:Metric,
                x_lbls = None,
                omit_empty = False,
                total_metric: Metric = None, # must if omit_empty
                coords_txt = False,
                h_lines = False,
                cut_trailing_zeros = False,
                omit_zero_lbls = False,
                add_avrg_line = False,
                bar_kwrgs = {}
                ):
    if omit_empty:
        total_arr = total_metric.val
        non_zero_indx = [i for i,v in enumerate(total_arr) if v > 0]
        m = data_metric[0]
        v_arr = m.val
        v_arr = v_arr[non_zero_indx]
        m = Metric(v_arr,'',None,m.flags)
        data_metric = m
        x_lbls=non_zero_indx
    
    actors = _mk_bar_plot(ax, 
                        data=data_metric,
                        x_lbls=x_lbls,
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

def plot_multi_step(
            ax:plt.axes, 
            data_container:EvaluationResultContainer,
            metric_temp : MetricTemplate,
            steps: list[str],
            colors: list[str]|None = None,
            x_lbls = None,
            omit_empty = False,
            coords_txt = False,
            h_lines = False,
            cut_trailing_zeros = False,
            omit_zero_lbls = False,
            add_avrg_line = False,
            ):
    
    total_metric = None
    if omit_empty:
        flgs = Flags.group_flags(metric_temp.flags)
        flgs[Flags.MetricFlags] = [Flags.MetricFlags.TOTAL]
        flgs = list(flgs.values())
        flgs = [f[0] for f in flgs]
        total_metric = data_container.get_data(MetricTemplate(flags=flgs))[0]

    if colors is None:
        colors = [None]*len(steps)
    from experiments.evaluation_statistics import aggregate_by_exp_type
    for step,color in zip(steps,colors):
        step_data = data_container.filter_by_exp_name(r'.+'+str(step))
        step_data = aggregate_by_exp_type(step_data)
        step_metrics:list[Metric] = step_data.get_data(metric_temp)

        
        alpha = 0.5

        bar_kwrgs={
                "label": str(step),
                "alpha": alpha,
                "color": color
                }
        if step == steps[0]:
            # highest epoch
            bar_kwrgs['alpha'] = alpha
            bar_kwrgs['color'] = color
            mk_bar_plt(ax,
                    data_metric=step_metrics,
                    x_lbls=x_lbls,
                    cut_trailing_zeros=cut_trailing_zeros,
                    omit_zero_lbls=omit_zero_lbls,
                    omit_empty=omit_empty,
                    total_metric=total_metric,
                    coords_txt=coords_txt,
                    h_lines=h_lines,
                    add_avrg_line=add_avrg_line,
                    bar_kwrgs= bar_kwrgs
                    )
        else:
            bar_kwrgs['alpha'] = alpha
            bar_kwrgs['color'] = color
            mk_bar_plt(ax,
                    data_metric=step_metrics,
                    x_lbls=x_lbls,
                    cut_trailing_zeros=cut_trailing_zeros,
                    omit_zero_lbls=omit_zero_lbls,
                    omit_empty=omit_empty,
                    total_metric=total_metric,
                    bar_kwrgs= bar_kwrgs,
                    )
            

def mk_e1_fig(res_path:str):
    E1 = EvaluationResultContainer.from_json(res_path)
    #TL ACC AVRG NO_ORACLE
    acc_tl_avrg = MetricTemplate(flags=[
        Flags.LevelFlags.TL,
        Flags.MetricFlags.ACC,
        Flags.DistrFlags.Avrg,
        Flags.PredictionFlags.NO_ORACLE,
    ])
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Experiment 1: Token Level Accuracy")
    ax.set_ylabel("Accuracy in %")
    ax.set_xlabel("Percentage of data used for training")
    x_lbls = ['1%','2%','4%','8%','16%','32%','64%','100%']
    # change this to choose epochs to show
    epochs_to_show = ['70','50','5']

    plot_multi_step(ax =ax,
                    data_container = E1,
                    metric_temp = acc_tl_avrg,
                    steps = epochs_to_show,
                    colors=None,
                    x_lbls = x_lbls,
                    h_lines = True,
                    coords_txt = True,
                    )
    ax.legend()
    return fig2, ax


def mk_e2_fig_1(res_path:str):
    E2 = EvaluationResultContainer.from_json(res_path)
    fig1, axs2_1 = plt.subplots(1, 2, figsize=(20, 10))
    axs2_1[0].set_title("Experiment 2: Token Level Accuracy\n by output length without oracle")
    axs2_1[0].set_ylabel("Accuracy in %")
    axs2_1[0].set_xlabel("Output length")

    axs2_1[1].set_title("Experiment 2: Token Level Accuracy\n by input length without oracle")
    axs2_1[1].set_ylabel("Accuracy in %")
    axs2_1[1].set_xlabel("Input length")

    # ACC TL NO ORACLE OUTLEN
    acc_tl_outlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.TL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.OutLen,
            Flags.PredictionFlags.NO_ORACLE,
            ])

    # ACC TL NO ORACLE INLEN
    acc_tl_inlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.TL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.InLen,
            Flags.PredictionFlags.NO_ORACLE,
            ])

    # change this to choose epochs to show
    epochs_to_show = ['70','40','1']
    colors = ["blue","red","yellow"]

    plot_multi_step(ax =axs2_1[0],
                    data_container = E2,
                    metric_temp = acc_tl_outlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    h_lines=True,
                    coords_txt=True
                    )

    plot_multi_step(ax =axs2_1[1],
                    data_container = E2,
                    metric_temp = acc_tl_inlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    omit_zero_lbls=True,
                    omit_empty= True,
                    h_lines=True,
                    coords_txt=True
                    )

    axs2_1[0].legend()
    axs2_1[1].legend()
    return fig1, axs2_1


def mk_e2_fig_2(res_path:str):
    E2 = EvaluationResultContainer.from_json(res_path)
    fig2, axs2_2 = plt.subplots(1, 2, figsize=(20, 10))
    axs2_2[0].set_title("Experiment 2: Token Level Accuracy\n by output length with oracle")
    axs2_2[0].set_ylabel("Accuracy in %")
    axs2_2[0].set_xlabel("Output length")
    axs2_2[1].set_title("Experiment 2: Token Level Accuracy\n by input length with oracle")
    axs2_2[1].set_ylabel("Accuracy in %")
    axs2_2[1].set_xlabel("Input length")

    # ACC TL ORACLE OUTLEN
    acc_tl_orac_outlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.TL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.OutLen,
            Flags.PredictionFlags.ORACLE,
            ])

    # ACC TL ORACLE INLEN
    acc_tl_orac_inlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.TL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.InLen,
            Flags.PredictionFlags.ORACLE,
            ])

    # change this to choose epochs to show
    epochs_to_show = [70,40,1]
    colors = ["blue","red","yellow"]

    plot_multi_step(ax =axs2_2[0],
                    data_container = E2,
                    metric_temp = acc_tl_orac_outlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    omit_zero_lbls=True,
                    h_lines=True,
                    coords_txt=True
                    )

    plot_multi_step(ax =axs2_2[1],
                    data_container = E2,
                    metric_temp = acc_tl_orac_inlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    omit_zero_lbls=False,
                    h_lines=True,
                    coords_txt=True
                    )

    axs2_2[0].legend()
    axs2_2[1].legend()
    return fig2, axs2_2


def mk_e2_fig_3(res_path:str):
    E2 = EvaluationResultContainer.from_json(res_path)
    fig3, axs2_3 = plt.subplots(1, 2, figsize=(20, 10))
    axs2_3[0].set_title("Experiment 2: Sequence Level Accuracy\n by output length with oracle")
    axs2_3[0].set_ylabel("Accuracy in %")
    axs2_3[0].set_xlabel("Output length")
    axs2_3[1].set_title("Experiment 2: Sequence Level Accuracy\n by input length with oracle")
    axs2_3[1].set_ylabel("Accuracy in %")
    axs2_3[1].set_xlabel("Input length")

    # ACC SL ORACLE OUTLEN
    acc_sl_orac_outlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.SL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.OutLen,
            Flags.PredictionFlags.ORACLE,
            ])

    # ACC SL ORACLE INLEN
    acc_sl_inlen = MetricTemplate(
        flags=[
            Flags.LevelFlags.SL,
            Flags.MetricFlags.ACC,
            Flags.DistrFlags.InLen,
            Flags.PredictionFlags.ORACLE,
            ])

    # change this to choose epochs to show
    epochs_to_show = [70,40,1]
    colors = ["blue","red","yellow"]
    plot_multi_step(ax =axs2_3[0],
                    data_container = E2,
                    metric_temp = acc_sl_orac_outlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    omit_zero_lbls=True,
                    h_lines=True,
                    coords_txt=True
                    )

    plot_multi_step(ax =axs2_3[1],
                    data_container = E2,
                    metric_temp = acc_sl_inlen,
                    steps = epochs_to_show,
                    colors=colors,
                    cut_trailing_zeros=True, 
                    omit_zero_lbls=False,
                    h_lines=True,
                    coords_txt=True
                    )

    axs2_3[0].legend()
    axs2_3[1].legend()
    return fig3, axs2_3
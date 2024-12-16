import matplotlib.pyplot as plt
import numpy as np
import re
from experiments.evaluation_result import EvaluationResult
from experiments.evaluation_result_container import EvaluationResultContainer
from experiments.metric import MetricTemplate, Flags
from experiments.evaluation_statistics import  aggregate_by_subname
from experiments.evaluation_statistics import aggregate_by_subname

def prepare_scalars_for_plot(
                      results:EvaluationResultContainer|EvaluationResult, 
                      template:MetricTemplate, 
                      xs = None,
                      x_lables = None,
                      x_lbls_re_patt = None,
                      ):
    data = results.get_data(template)
    if not x_lables and not x_lbls_re_patt:
        x_lables = np.arange(len(data))
    elif x_lbls_re_patt:
        x_lables = [re.search(x_lbls_re_patt, m.name).group(0) for m in data]
    if not xs :
        xs = np.arange(len(data))
    ys = [m.val for m in data]
    xtics = np.arange(len(data))
    return xs, ys, x_lables, xtics


def plot_k_fold_reps(ax,
                    rep1:EvaluationResultContainer,
                    rep2:EvaluationResultContainer,
                    rep3:EvaluationResultContainer,
                    template:MetricTemplate
                     ):

    # averages over three folds
    rep1_fld_avrg = aggregate_by_subname([rep1], id_sub_name="fold", id_range=(1, 3), reps=1)
    rep2_fld_avrg = aggregate_by_subname([rep2], id_sub_name="fold", id_range=(1, 3), reps=1)
    rep3_fld_avrg = aggregate_by_subname([rep3], id_sub_name="fold", id_range=(1, 3), reps=1)
    # averages over the three repetitions of the 3-fold cross validation
    rep_avrg_fld_avrg = aggregate_by_subname([rep1,rep2,rep3], id_sub_name="fold", id_range=(1, 3), reps=3)

    # prepare scalars
    xs_patt = r"(?<=epoch_)[0-9]+"

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep1_fld_avrg, template, x_lbls_re_patt=xs_patt)
    ax.plot(xs,ys,  linestyle='-', color='blue', label="repetition 1", alpha = 0.5)

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep2_fld_avrg, template, x_lbls_re_patt=xs_patt)
    ax.plot(xs,ys,  linestyle='-', color='green', label="repetition 2", alpha = 0.5)

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep3_fld_avrg, template, x_lbls_re_patt=xs_patt)
    ax.plot(xs,ys,  linestyle='-', color='brown', label="repetition 3", alpha = 0.5)

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep_avrg_fld_avrg, template, x_lbls_re_patt=xs_patt)
    ax.plot(xs,ys, marker='o', linestyle='-', color='orange', label="average")

    # Customize plot
    ax.set_xticks(xtics, x_lables)
    #axs[0].set_yscale('log')
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


def mk_fig(f3_r1, f3_r2, f3_r3):
    fig, axs = plt.subplots(5, 1, figsize=(15,35))
    ax_err_tok = axs[0]
    ax_err_or_tok = axs[1]
    ax_err_seq = axs[2]
    ax_err_or_seq = axs[3]
    ax_err_by_flags = axs[4]


    #ax_err_tok.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_tok, f3_r1, f3_r2, f3_r3,template=err_tok_temp)
    ax_err_tok.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_tok.__str__(),
                            fontsize = 10)

    #ax_err_or_tok.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_or_tok, f3_r1, f3_r2, f3_r3,template=err_or_tok_temp)
    ax_err_or_tok.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_or_tok.__str__(),
                            fontsize = 10)

    #ax_err_seq.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_seq, f3_r1, f3_r2, f3_r3,template=err_seq_temp)
    ax_err_seq.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_seq.__str__(),
                            fontsize = 10)

    #ax_err_or_seq.set_ylim(0, 0.015)
    plot_k_fold_reps(ax_err_or_seq, f3_r1, f3_r2, f3_r3,template=err_or_seq_temp)
    ax_err_or_seq.set_title("1 Avrg Error for k-fold cross validation over 3 repetitions\n"+err_or_seq.__str__(),
                            fontsize = 10)


    
    xs_patt = r"(?<=epoch_)[0-9]+"
    rep_avrg_fld_avrg = aggregate_by_subname([f3_r1, f3_r2, f3_r3], id_sub_name="fold", id_range=(1, 3), reps=3)

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep_avrg_fld_avrg,err_tok_temp, x_lbls_re_patt=xs_patt)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='green', label=str(err_tok))

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep_avrg_fld_avrg,err_or_tok_temp, x_lbls_re_patt=xs_patt)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='red', label=str(err_or_tok))

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep_avrg_fld_avrg,err_seq_temp, x_lbls_re_patt=xs_patt)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='orange', label=str(err_seq))

    xs, ys, x_lables, xtics = prepare_scalars_for_plot(rep_avrg_fld_avrg,err_or_seq_temp, x_lbls_re_patt=xs_patt)
    ax_err_by_flags.plot(xs,ys,  linestyle='-', color='purple', label=str(err_or_seq))


    # xtics = axs[0].get_xticklabels()
    # xtics = [t.get_text() for t in xtics]
    ax_err_by_flags.set_xticks(xtics,x_lables)

    ax_err_by_flags.set_xlabel("Epoch number")
    ax_err_by_flags.set_ylabel("Error (%)")
    ax_err_by_flags.legend()

    return fig, axs



def mk_bar_plot(ax:plt.axes, 
                data:EvaluationResultContainer,
                temp:MetricTemplate,
                x_lbls= None,
                coords_txt = True,
                h_lines = True,
                bar_kwrgs = None
                ):
    xs, ys, x_lables, xtics = prepare_scalars_for_plot(results= data*100,
                                                   template= temp, 
                                                   x_lables=x_lbls,)                      
    # ys.append(ys[0])
    # ys = ys[1:]
    points = len(ys)

    if bar_kwrgs is None:
        bar_kwrgs = {}

    ax.bar(xs, ys, **bar_kwrgs)
    if h_lines:
        ax.hlines([20,40,60,80,100],xmin=-0.5, xmax=points-0.5, colors = "grey", alpha = 0.5)
    if coords_txt:
        for i in range(len(ys)):
            ax.text(xs[i], ys[i], f"{ys[i]:.2f}", ha='center', va='bottom')
    ax.set_xlim(-0.5,points-0.5)
    ax.set_xticks(xtics)
    ax.set_xticklabels(x_lables)
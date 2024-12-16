from experiments.evaluation_result_container import EvaluationResultContainer
from experiments.experiment_type import ExperimentType
from experiments.metric import MetricTemplate
import numpy as np

def aggregate_by_subname(containers:list[EvaluationResultContainer], id_sub_name:str, id_range:tuple, reps = 1):
    """
    Aggregates evaluation results by a specified subname and range.
    First calculates the average of the repetitions (if any) and then groups
    the evaluation results by the specified subname and range.

    Args:
        containers (list[EvaluationResultContainer]): A list of EvaluationResultContainer objects to aggregate.
        id_sub_name (str): The subname identifier to filter the evaluation results (e.g., "fold", "epoch").
        id_range (tuple): A tuple specifying the range of IDs to consider. It can be (min_id, max_id) or (min_id, max_id, step).
        reps (int, optional): The number of repetitions of the experiment to average over. Defaults to 1.

    Returns:
        EvaluationResultContainer: An EvaluationResultContainer object containing the aggregated evaluation results.

    Examples:
        Experiment run 3 times with 3 folds and 100 epochs.
        The first version finds the average of the repetitions and then average of epoches. Finally groups by fold.
        aggregate_evaluation_by_subname(
            [e1_f3_r1, e1_f3_r2, e1_f3_r3], reps=3, id_sub_name="fold", id_range=(1, 3))
        The second averages by fold and groups by epoch.
        aggregate_evaluation_by_subname(
            [e1_f3_r1, e1_f3_r2, e1_f3_r3], reps=3, id_sub_name="epoch", id_range=(5, 100, 5))
    """
    min_id = id_range[0]
    max_id = id_range[1]
    if len(id_range) == 2:
        step = 1
    else:
        step = id_range[2]

    if reps > 1:
        containers = np.mean(containers, axis=0)
    
    containers = EvaluationResultContainer(containers)
    
    # divide by subname
    divided = []
    g_id = min_id 
    exp = id_sub_name
    while True:
        exp_id = r'.+'+ f"{exp}_{g_id}"+ r'[^0-9]' 
        sub_group = containers.filter_by_exp_name(exp_id)
        if len(sub_group) == 0:
            break
        divided.append(sub_group)
        g_id += step
        # print(f"sub_group_id: {exp_id}, len: {len(sub_group)}")
        # [print(r) for r in sub_group]
        if g_id > max_id:
            break

    avrg = np.mean(divided, axis=0)
    # avrg = divided[0]
    # for i in range(1, len(divided)):
    #     avrg += divided[i]
    # avrg /= len(divided)
    return EvaluationResultContainer(avrg)



def aggregate_by_exp_type(containers:list[EvaluationResultContainer]):
    """
    Aggregates evaluation results by experiment type.
    Args:
        containers (list[EvaluationResultContainer]): A list of EvaluationResultContainer objects to aggregate.
    Returns:
        EvaluationResultContainer: An EvaluationResultContainer object containing the aggregated evaluation results.
    """
    # get unique experiment types
    d = {}
    types = [d.update({e.experiment_type:None}) for e in containers]
    types = list(d.keys())
    types.sort()
    avrgs = []
    for e_type in types:
        sub_group = containers.filter_by_exp_type(e_type)
        avrg = np.mean(sub_group, axis=0)
        avrgs.append(avrg)
    return EvaluationResultContainer(avrgs)
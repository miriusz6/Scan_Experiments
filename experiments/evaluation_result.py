from __future__ import annotations
from experiments.experiment_type import ExperimentType
from experiments.metric import Metric, MetricTemplate, Flags
from experiments.utils import merge_names
import numpy as np
import operator


class EvaluationResult:
    @staticmethod
    def _mk_lookup():
        f_groups = Flags.All
        # all flags by group
        all_f_grouped = [f.All for f in f_groups]
        # number of metrics in whole lookup
        metrics_numb = np.prod([len(f) for f in all_f_grouped])
        # create grid of all possible combinations
        # of flags, each column is a flag group,
        # last column is index of metric
        grid = np.meshgrid(*all_f_grouped)
        grid = [np.ravel(x) for x in grid]
        grid = np.array(grid)
        print(grid.shape)
        grid = np.array(grid)
        indxs = np.arange(metrics_numb)
        grid = np.vstack([grid,indxs])
        return grid
    
    lookup = _mk_lookup()

    def __init__(self, data: list[Metric]|dict, e_name:str = "Mixed" , e_type = ExperimentType.MIX):
        self.experiment_type = e_type
        self.experiment_name = e_name
        if isinstance(data, list):
            self.data = data
        elif isinstance(data, dict):
            self.data = self._unpack_data(data)
        else:
            self.data = [0]* self.lookup.shape[1]

    @classmethod
    def _find_index(cls, template: MetricTemplate):
        f_grouped = Flags.group_flags(template.flags)
        # if no flags in a group, add all flags of that group
        [f_grouped.update({k:k.All}) if len(v) == 0 else None for k,v in f_grouped.items()]
        
        indxs = np.full(cls.lookup.shape[1], True)
        for i,flags_in_group in enumerate(f_grouped.values()):
            hits = np.isin(cls.lookup[i],flags_in_group, assume_unique=True)
            indxs = indxs & hits
        return cls.lookup[-1][indxs]
    
    def get_data(self, template: MetricTemplate):
        indxs = self._find_index(template)
        return [self.data[i] for i in indxs]
    
    def _unpack_data(self,data):
        
        from experiments.utils import switch_dict_lvls, flatten_dict
        # D L M P
        groups = Flags.All
        curr_d_lvl = data
        # P M D L
        order = []
        for _ in groups:
            k = list(curr_d_lvl.keys())[0]
            order.append(Flags.get_group(k))
            curr_d_lvl = curr_d_lvl[k]
        # 3 2 0 1
        proper_order = [order.index(g) for g in groups]
        
        data = switch_dict_lvls(data, proper_order)
        
        data = flatten_dict(data)
        
        return [Metric(v,self.experiment_name,self.experiment_type,list(k)) for k,v in data.items()]
        
        
    
    
    def print(self, scalars_only = True):
        if scalars_only:
            lst = self.get_data(MetricTemplate(flags=[Flags.DistrFlags.Avrg]))
        else:
            lst = self.data
        print(self.__str__())
        list(map(print,lst))

    def _update_info(self, other:EvaluationResult, oper):
        if isinstance(other, EvaluationResult):
            new_data = [oper(a, b) for a,b in zip(self.data, other.data)]
            new_name = merge_names(self.experiment_name, other.experiment_name)
            if other.experiment_type == self.experiment_type:
                return EvaluationResult(new_data, new_name, self.experiment_type)
            return EvaluationResult(new_data, new_name)
        else:
            new_data = [oper(a,other) for a in self.data]
            return EvaluationResult(new_data, self.experiment_name, self.experiment_type)

    def __str__(self):
        return f"Experiment: {self.experiment_name} Results, Type: {self.experiment_type.name}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self._update_info(other, operator.add)
    
    def __sub__(self, other):
        return self._update_info(other, operator.sub)
    
    def __mul__(self, other):
        return self._update_info(other, operator.mul)
    
    def __truediv__(self, other):
        return self._update_info(other, operator.truediv)
    
    def __floordiv__(self, other):
        return self._update_info(other, operator.floordiv)
    
    def __mod__(self, other):
        return self._update_info(other, operator.mod)
    
    def __pow__(self, other):
        return self._update_info(other, operator.pow)
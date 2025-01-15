from __future__ import annotations
from experiments.experiment_type import ExperimentType
from experiments.metric import Metric, MetricTemplate, Flags, Flag
from experiments.metric_utils import merge_names
from experiments.metric_utils import flatten_dict
import numpy as np
import operator
from torch.nn.utils import clip_grad_norm_

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

    def __init__(self, data: list[Metric]|dict, e_name:str = ExperimentType.UNDEF.name , e_type = ExperimentType.UNDEF):
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
        ret = []
        for i in indxs:
            m = self.data[i]
            # if is a metric
            if isinstance(m, Metric):
                ret.append(m)
        return ret
    
    def set_data(self, metrics : list[Metric]):
        e_type = self.experiment_type
        for m in metrics:
            e_type = ExperimentType.UNDEF if e_type != m.e_type else e_type
            indx = self._find_index(m.to_template())
            self.data[indx] = m
        self.experiment_type = e_type
        self.experiment_name = ExperimentType.UNDEF.name

    def rename(self, new_name:str):
        self.experiment_name = new_name

    def _unpack_data(self,data):
        data = flatten_dict(data)
        ret = [0]*self.lookup.shape[1]
        for k,v in data.items():
            key_lst = list(k)
            m = Metric(v,self.experiment_name,self.experiment_type,[Flag(x) for x in key_lst])
            t = MetricTemplate(flags= [Flag(x) for x in key_lst])
            indx = self._find_index(t)[0]
            ret[indx] = m
        return ret
     
        
    def print(self, scalars_only = True):
        if scalars_only:
            lst = self.get_data(MetricTemplate(flags=[Flags.DistrFlags.Avrg, Flags.MetricFlags.ACC]))
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
        
    @classmethod
    def to_dict(cls, eval_result: EvaluationResult):
        d = {}
        for i,m in enumerate(eval_result.data):
            if isinstance(m, Metric):
                d[i] = Metric.to_dict(m)
            else:
                d[i] = m
        return {"e_name": eval_result.experiment_name, "e_type": eval_result.experiment_type.value, "data": d}
    
    @classmethod
    def from_dict(cls,d: dict):
        data = d["data"]
        data = [Metric.from_dict(v) if isinstance(v, dict) else v for v in data.values()]
        e_name = d["e_name"]
        e_type = ExperimentType( d["e_type"])
        return EvaluationResult(data, e_name, e_type)
    
    def to_template(self):
        return MetricTemplate(flags=[m.flags for m in self.data if isinstance(m, Metric)])

    def __str__(self):
        return f"Experiment: {self.experiment_name} Results, Type: {self.experiment_type.name}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return self._update_info(other, operator.add)
    
    def __sub__(self, other):
        return self._update_info(other, operator.sub)
    
    def __rsub__(self, other):
        return self._update_info(other, operator.sub)
    
    def __radd__(self, other):
        return self._update_info(other, operator.add)
    
    

    
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
    
    def __and__(self, other):
        return self._update_info(other, operator.and_)
    
    def __or__(self, other):
        return self._update_info(other, operator.or_)
    
    def __xor__(self, other):
        return self._update_info(other, operator.xor)
    
    def __lshift__(self, other):
        return self._update_info(other, operator.lshift)
    
    def __rshift__(self, other):
        return self._update_info(other, operator.rshift)
    
    def __neg__(self):
        return EvaluationResult([-x for x in self.data], self.experiment_name, self.experiment_type)
    
    def __pos__(self):
        return EvaluationResult([+x for x in self.data], self.experiment_name, self.experiment_type)
    
    def __abs__(self):
        return EvaluationResult([abs(x) for x in self.data], self.experiment_name, self.experiment_type)
    

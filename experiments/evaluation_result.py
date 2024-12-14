from __future__ import annotations
from experiments.experiment_type import ExperimentType
from experiments.metric import Metric, MetricFlag, MetricType, OracleFlag, MetricTemplate
from experiments.utils import merge_names
import numpy as np
import operator


class EvaluationResult:
    @staticmethod
    def _mk_lookup():
        oracle_flags = len(OracleFlag)
        flags = len(MetricType)
        metrics = len(MetricFlag)
        metrics_numb = oracle_flags * flags * metrics
        lookup = np.empty((4,metrics_numb), dtype=object)
        o = flags * metrics
        for i,of in enumerate(OracleFlag):
            p1 = lookup[:,i*o:(i+1)*o]
            p1[0,:] = of
            # second column
            for j,m in enumerate(MetricType):
                p2 = p1[:,j*metrics:(j+1)*metrics]
                p2[1,:] = m
                # third column
                for k,f in enumerate(MetricFlag):
                    p2[2,k] = f

            lookup[:,i*o:(i+1)*o] = p1
        lookup[3,:] =  np.arange(0,metrics_numb)
        return lookup
    
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
        oracle_flags = template.o_flags
        metric_types = template.m_types
        metric_flags = template.m_flags
        
        o_indxs = np.ones(cls.lookup[0].shape, dtype=bool)
        f_indxs = np.ones(cls.lookup[1].shape, dtype=bool)
        t_indxs = np.ones(cls.lookup[2].shape, dtype=bool)
        
        if oracle_flags is not None:
            o_indxs = np.isin(cls.lookup[0],oracle_flags, assume_unique=True)
        if metric_types is not None:
            t_indxs = np.isin(cls.lookup[1],metric_types, assume_unique=True)
        if metric_flags is not None:
            f_indxs = np.isin(cls.lookup[2],metric_flags, assume_unique=True)
        indxs = np.where(o_indxs & f_indxs & t_indxs, True, False)
        indxs = cls.lookup[3][indxs]
        return indxs
    
    def get_data(self, template: MetricTemplate):
        indxs = self._find_index(template)
        return [self.data[i] for i in indxs]
    
    def _unpack_data(self,data):
        d = []
        e = self.experiment_type
        en = self.experiment_name
        for of in OracleFlag:
            for m in MetricType:
                for f in MetricFlag:
                    data_point = Metric(data[of][m][f],en,e, of, f, m)
                    d.append(data_point)
        return d
    def print(self, scalars_only = True):
        if scalars_only:
            fs = MetricFlag.get_scalar_flags()
            lst = self.get_data(MetricTemplate(m_flags=fs))
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
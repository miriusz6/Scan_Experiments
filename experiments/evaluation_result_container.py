from __future__ import annotations
from experiments.evaluation_result import EvaluationResult
from experiments.metric import MetricTemplate
import re
from collections import OrderedDict
import json
from experiments.experiment_type import ExperimentType

class EvaluationResultContainer():
    def __init__(self, e_results:list[EvaluationResult|EvaluationResultContainer] = None):
        self.results_map = OrderedDict()
        # self.results = []
        # self.result_names = []
        if e_results is not None:
            for e in e_results:
                if isinstance(e, EvaluationResult):
                    self.append_results(e)
                else:
                    self.merge_containers(e)
    
    
    @property
    def result_names(self):
        return list(self.results_map.keys())
    
    @property
    def results(self):
        return list(self.results_map.values())
    
    def merge_containers(self, other:EvaluationResultContainer):
        for e in other.results:
            self.append_results(e)

    def append_results(self,e: EvaluationResult):
        i = 0
        n = e.experiment_name
        while n in self.results_map:
            n = f"{e.experiment_name}({i})"
            i += 1
        e.experiment_name = n

        self.results_map[e.experiment_name] = e
        # self.results = list(self.results_map.values())
        # self.result_names = list(self.results_map.keys())

    def get_data(self,template: MetricTemplate):
        experiment_types = template.e_types
        data = []
        if experiment_types is not None:
            hits = [r for r in self.results_map.values() if r.experiment_type in experiment_types]
            for r in hits:
                data += r.get_data(template)
            return data
        else:
            for e in self.results_map.keys():
                data += self.results_map[e].get_data(template)
        return data
    
    def filter_by_exp_name(self, pattern:str):
        matches = [re.match(pattern,r.experiment_name,re.RegexFlag.DOTALL) for r in self.results]
        ret = []
        for i,m in enumerate(matches):
            if m:
                ret.append(self.results[i])
        return EvaluationResultContainer(ret)
    
    

    
    def filter_by_exp_type(self, e_types:ExperimentType|list[ExperimentType]):
        if isinstance(e_types, ExperimentType):
            e_types = [e_types]
        return EvaluationResultContainer([r for r in self.results if r.experiment_type in e_types])

    
    
    def rename_element(self, old_name:str, new_name:str):
        if old_name in self.results_map and new_name not in self.results_map:
            self.results_map[old_name].rename(new_name)
            self.results_map[new_name] = self.results_map.pop(old_name)
            # self.result_names = list(self.results_map.keys())
            # self.results = list(self.results_map.values())
    
    
    def pop(self,to_pop:int|str):
        if isinstance(to_pop, int):
            key = self.result_names[to_pop]
        else:
            key = to_pop
        return self.results_map.pop(key)
    
    # def append(self, e:EvaluationResult):
    #     self.append_results(e)
    
    def sort_by_exp_names(self):
        self.results_map = OrderedDict(sorted(self.results_map.items(), key=lambda x: x[0]))
        #self.results_map.items().sort(key=lambda x: x[0])
        # self.result_names = list(self.results_map.keys())
        # self.results = list(self.results_map.values())
        
    def sort_by_exp_types(self):
        self.results_map = OrderedDict(sorted(self.results_map.items(), key=lambda x: x[1].experiment_type.value))
        


    @classmethod
    def from_dict(cls, d:dict):
        results = [EvaluationResult.from_dict(v) for v in d["results"]]
        return EvaluationResultContainer(results)
    
    @classmethod
    def to_dict(cls, container:EvaluationResultContainer):
        results = [EvaluationResult.to_dict(v) for v in container.results]
        return {"results": results}

    @classmethod
    def from_json(cls, path:str):
        with open(path, "r") as f:
            res_d = json.load(f)
        return cls.from_dict(res_d)
    
    @classmethod
    def to_json(cls, container:EvaluationResultContainer, path:str):
        with open(path, "w") as f:
            json.dump(cls.to_dict(container), f, indent=4)
        
    def __str__(self):
        s = "Experiment Results Container:\n"
        for r in self.results:
            s += str(r) + "\n"
        return s
    
    def __repr__(self):
        return self.__str__()
    
    # make iterable
    def __iter__(self):
        return iter(self.results)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.results[key]
        else:
            return EvaluationResultContainer(self.results[key])
    
    def __len__(self):
        return len(self.results)
    
    def __add__(self, other):
        if isinstance(other, EvaluationResultContainer):
            new_res = [ x + y for x,y in zip(self.results, other.results)]
            return EvaluationResultContainer(new_res)
        else:
            return EvaluationResultContainer([r + other for r in self.results])
    
    def __sub__(self, other):
        if isinstance(other, EvaluationResultContainer):
            new_res = [ x - y for x,y in zip(self.results, other.results)]
            return EvaluationResultContainer(new_res)
        else:
            return EvaluationResultContainer([r - other for r in self.results])
        
    def __mul__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r * o for r,o in zip(self.results, other.results)])
        else:
            return EvaluationResultContainer([r * other for r in self.results])
    
    def __truediv__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r / o for r,o in zip(self.results, other.results)])
        else:
            return EvaluationResultContainer([r / other for r in self.results])
        
    def __floordiv__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r // o for r,o in zip(self.results, other.results)])
        else:
            return EvaluationResultContainer([r // other for r in self.results])
        
    def __mod__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r % o for r,o in zip(self.results, other.results)])
        else:
            return EvaluationResultContainer([r % other for r in self.results])
        
    def __pow__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r ** o for r,o in zip(self.results, other.results)])
        else:
            return EvaluationResultContainer([r ** other for r in self.results])
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __rfloordiv__(self, other):
        return self.__floordiv__(other)
    
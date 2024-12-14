from __future__ import annotations
from experiments.evaluation_result import EvaluationResult
from experiments.metric import MetricTemplate



class EvaluationResultContainer():
    def __init__(self, e_results:list[EvaluationResult|EvaluationResultContainer] = None):
        self.results_map = {}
        self.results = []
        self.result_names = []
        if e_results is not None:
            for e in e_results:
                if isinstance(e, EvaluationResult):
                    self.append_results(e)
                else:
                    self.merge_containers(e)
    
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
        self.results.append(e)
        self.result_names.append(e.experiment_name)

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


    def __str__(self):
        return f"Experiment Results Container: {self.result_names}"
    
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
            return EvaluationResultContainer(self.results + other.results)
        else:
            return EvaluationResultContainer(self.results + [other])
    
    def __sub__(self, other):
        if isinstance(other, EvaluationResultContainer):
            return EvaluationResultContainer([r for r in self.results if r not in other.results])
        else:
            return EvaluationResultContainer([r for r in self.results if r != other])
        
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
    
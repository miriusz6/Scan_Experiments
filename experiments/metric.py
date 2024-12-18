from __future__ import annotations
from enum import IntEnum, auto
import operator
from experiments.experiment_type import ExperimentType
from experiments.metric_utils import merge_names
import numpy as np
from experiments.metric_utils import remove_mods, remove_vars


class Flag(IntEnum):
    # Metric Type
    CORRECT = auto()
    TOTAL = auto()
    ACC = auto()
    ERR = auto()
    # Level Type
    TL = auto()
    SL = auto()
    # Distr Type
    InLen = auto()
    OutLen = auto()
    Avrg = auto()
    Sum = auto()
    # Prediction Type
    ORACLE = auto()
    NO_ORACLE = auto()

    @staticmethod
    def get__all_flags():
        return [f for f in Flag]

    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class FlagSubClass():
    _all = []

    def belong(self, flag: Flag):
        return flag in self._all
    @property
    def All(self):
        self._all.sort(key=lambda x: x.value)
        return self._all

class _PredictionFlags(FlagSubClass):
    ORACLE = Flag.ORACLE
    NO_ORACLE = Flag.NO_ORACLE
    _all = [ORACLE, NO_ORACLE]

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return self.__str__()

class _MetricFlags(FlagSubClass):
    CORRECT = Flag.CORRECT
    TOTAL = Flag.TOTAL
    ACC = Flag.ACC
    ERR = Flag.ERR
    _all = [CORRECT, TOTAL, ACC, ERR]

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return self.__str__()

class _LevelFlags(FlagSubClass):
    TL = Flag.TL
    SL = Flag.SL
    _all = [TL, SL]

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return self.__str__()
    
class _DistrFlags(FlagSubClass):
    InLen = Flag.InLen
    OutLen = Flag.OutLen
    Avrg = Flag.Avrg
    Sum = Flag.Sum
    _all = [InLen, OutLen, Avrg, Sum]

    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return self.__str__()
    

class _Flags():
    PredictionFlags = _PredictionFlags()
    MetricFlags = _MetricFlags()
    LevelFlags = _LevelFlags()
    DistrFlags = _DistrFlags()
    _all = [PredictionFlags, MetricFlags, LevelFlags, DistrFlags]
    
    def group_flags(self, flags: list[Flag]):
        ret = {}
        groups = self._all
        for g in groups:
            ret[g] = [f for f in flags if f in g.All]
        return ret
    
    def get_group(self, flag: Flag):
        for g in self._all:
            if g.belong(flag):
                return g
        return None
    
    @property
    def All(self):
        self._all.sort(key=str)
        return self._all
    
    def __str__(self):
        s = "Avaiable Flag Groups: \n"
        for f in self._all:
            s += f"{f}\n"
        return s
    
    def __repr__(self):
        return self.__str__()
        
Flags = _Flags()



class MetricTemplate():
    def __init__(self,
                 e_types: list[ExperimentType] = None,
                 flags: list[Flag] = None):
        self.e_types = e_types
        self._flags = flags#.sort(key=lambda x: x.value)
    
    @property
    def flags(self):
        return self._flags
    
    @property
    def flagsS(self):
        s = ""
        [s.join(f"{f.name} ") for f in self._flags]
        return s
    
    def __str__(self):
        return self.flagsS
    
    def __repr__(self):
        return self.__str__()

class Metric():
    print_mods = True
    print_vars = True
    def __init__(self, val, e_name:str, e_type: ExperimentType, flags: list[Flag]):
        self._val = val
        self.e_type = e_type
        self.e_name = e_name
        if not self.e_name.startswith("(") and not self.e_name == ExperimentType.UNDEF.name: 
            self.e_name = f"(ORG){self.e_name}"
        self._flags = flags
        f_groups = Flags.group_flags(flags)
        
        # Check if only one flag of each group is present
        single_flags = [len(vals) == 1 for vals in f_groups.values()]
        if sum(single_flags) != len(f_groups.values()):
            raise ValueError("Only one flag of each group is _allowed")
        # self._flags = flags
        self._flags.sort(key=lambda x: x.value)

    @property
    def val(self):
        return self._val
    
    @val.setter
    def val(self, v):
        if not self.e_name.startswith("(") and not self.e_name == ExperimentType.UNDEF.name: 
            self.e_name = f"(MOD){self.e_name}"
        self._val = v

    @property
    def flags(self):
        return self._flags
    
    @property
    def flagsS(self):
        return " ".join([f"{f.name} " for f in self.flags])
    
    # def get_name(mods = False, vars = False):
    #     if mods:
    #         return self.e_name
    #     if vars:
    #         return self.e_name[1:]
    #     return self.e_name
    
    @classmethod
    def to_dict(cls, metric:Metric):
        v = metric.val
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return {"e_name": metric.e_name, "e_type": metric.e_type.value, "flags": metric.flags, "val": v}
    
    @classmethod
    def from_dict(cls,d: dict):
        v = d["val"]
        if isinstance(v, list):
            v = np.array(v)
        return Metric(v, d["e_name"], ExperimentType(d["e_type"]), [Flag(f) for f in d["flags"]])


    @property
    def name(self):
        return self.__str__()
    
    def to_template(self):
        return MetricTemplate(flags=[self.flags])
    

    def __str__(self):
        name = self.e_name
        #if cls.print_mods and cls.print_vars:
        if not Metric.print_vars: 
            name = remove_vars(name)    
        if not Metric.print_mods:
            name = remove_mods(name)
        return f"{name} {self.e_type.name} {self.flagsS} : {self.val}"


    
    def __repr__(self):
        return self.__str__()

    def _update_modifiers(self, name, new_mod:str):
        i = name.find(")")
        mods = name[1:i]
        if mods == "ORG":
            return f"({new_mod}"+name[i:]
        return '(' + new_mod +','+  name[1:]

    def __lt__(self, other):
        if isinstance(other, Metric):
            return self.val < other.val
        return self.val < other
    
    def __le__(self, other):
        if isinstance(other, Metric):
            return self.val <= other.val
        return self.val <= other
    
    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.val == other.val
        return self.val == other
    
    def __ne__(self, other):
        if isinstance(other, Metric):
            return self.val != other.val
        return self.val != other
    
    def __gt__(self, other):
        if isinstance(other, Metric):
            return self.val > other.val
        return self.val > other
    
    def __ge__(self, other):
        if isinstance(other, Metric):
            return self.val >= other.val
        return self.val >= other
    
    def _smth(self,other, oper, mod:str):
        if not isinstance(other, Metric):
            new_name = self._update_modifiers(self.e_name, mod)
            new_val = oper(self.val, other)
            return Metric(new_val, new_name, self.e_type, self.flags)
        new_name = merge_names(self.e_name, other.e_name)
        new_name = self._update_modifiers(new_name, mod)

        if isinstance(self.val, np.ndarray) and isinstance(other.val, np.ndarray):
            # pad both to tthe same length
            if len(self.val) < len(other.val):
                self.val = np.pad(self.val, (0, len(other.val) - len(self.val)))
            elif len(self.val) > len(other.val):
                other.val = np.pad(other.val, (0, len(self.val) - len(other.val)))
                
        new_val = oper(self.val, other.val)
        if other.e_type == self.e_type:
            new_type = self.e_type
        else:
            new_type = ExperimentType.UNDEF
        return Metric(new_val, new_name, new_type, self.flags)    
    
    def __add__(self, other):
        return self._smth(other, operator.add, "ADD")
    
    def __sub__(self, other):
        return self._smth(other, operator.sub, "SUB")
    
    def __mul__(self, other):
        return self._smth(other, operator.mul, "MUL")
    
    def __truediv__(self, other):
        return self._smth(other, operator.truediv, "DIV")
    
    def __floordiv__(self, other):
        return self._smth(other, operator.floordiv, "DIV")
    
    def __mod__(self, other):
        return self._smth(other, operator.mod, "MOD")
    
    def __pow__(self, other):
        return self._smth(other, operator.pow, "POW")

        
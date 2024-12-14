from enum import Enum, auto
import operator
from experiments.experiment import ExperimentType
from experiments.utils import merge_names

class OracleFlag(Enum):
    ORACLE = auto()
    NO_ORACLE = auto()

class MetricType(Enum):
    CORRECT = auto()
    TOTAL = auto()
    ACC = auto()
    ERR = auto()

class MetricFlag(Enum):
    TL = auto()
    SL = auto()
    TLIL = auto()
    SLIL = auto()
    TLOL = auto()
    SLOL = auto()
    
    @classmethod
    def get_scalar_flags(cls):
        return [cls.TL, cls.SL]
    @classmethod
    def get_array_flags(cls):
        return [cls.TLIL, cls.SLIL, cls.TLOL, cls.SLOL]
    

class Flag():
    OracleFlag = OracleFlag
    Metric = MetricFlag
    MetricType = MetricType

F = Flag()

class MetricTemplate():
    def __init__(self,
                 e_types: list[ExperimentType] = None,
                 o_flags: list[OracleFlag] = None,
                 m_flags: list[MetricFlag] = None,
                 m_types: list[MetricType] = None):
        self.e_types = e_types
        self.o_flags = o_flags
        self.m_flags = m_flags
        self.m_types = m_types
    
    @property
    def flags(self):
        return {self.o_flags, self.m_flags, self.m_types}
    
    @property
    def flagsS(self):
        s = ""
        s += [s.join(f.name) for f in self.o_flags]
        s += [s.join(f.name) for f in self.m_flags]
        s +=[s.join(f.name) for f in self.m_types]
        return s
    
    def __str__(self):
        return self.flagsS
    
    def __repr__(self):
        return self.__str__()

class Metric():
    def __init__(self, val, e_name:str, e_type: ExperimentType, o_flag: OracleFlag, m_flag: MetricFlag, m_type: MetricType):
        self._val = val
        self.e_type = e_type
        if not e_name.startswith("(") and not e_name == "MIXED": 
            e_name = f"(ORG){e_name}"
        self.e_name = e_name
        self.o_flag = o_flag
        self.m_flag = m_flag
        self.m_type = m_type

    @property
    def val(self):
        return self._val

    @property
    def flags(self):
        return (self.o_flag, self.m_flag, self.m_type)
    
    @property
    def flagsS(self):
        return f"{self.o_flag.name} {self.m_flag.name} {self.m_type.name}"


    def __str__(self):
        return f"{self.e_name} {self.e_type.name} {self.m_type.name} {self.m_flag.name} {self.o_flag.name}: {self.val}"
    
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
            return Metric(new_val, new_name, self.e_type, self.o_flag, self.m_flag, self.m_type)
        new_name = merge_names(self.e_name, other.e_name)
        new_name = self._update_modifiers(new_name, mod)
        new_val = oper(self.val, other.val)
        if other.e_type == self.e_type:
            new_type = self.e_type
        else:
            new_type = ExperimentType.MIX
        return Metric(new_val, new_name, new_type, self.o_flag, self.m_flag, self.m_type)
        
    
    def __add__(self, other):
        return self._smth(other, operator.add, "ADD")
    
    def __sub__(self, other):
        return self._smth(other, operator.sub, "SUB")
    
    def __mul__(self, other):
        return self._smth(other, operator.mul, "MUL")
    
    def __truediv__(self, other):
        return self._smth(other, operator.truediv, "DIV")
    
    def __floordiv__(self, other):
        return self._smth(other, operator.floordiv, "FDIV")
    
    def __mod__(self, other):
        return self._smth(other, operator.mod, "MOD")
    
    def __pow__(self, other):
        return self._smth(other, operator.pow, "POW")
    
        
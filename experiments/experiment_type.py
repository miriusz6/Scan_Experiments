from enum import IntEnum,auto
import os 
CURR_DIR = os.getcwd() #os.path.dirname(os.path.realpath(__file__))

# WORK IN PROGRESS
class ExperimentType(IntEnum):
    # Exper 1 simple split 80/20
    # Exper 1 size variations for p in {1,2,4,8,16,32,64}
    # Exper 2: seq length differs between train and test
    # Exper 3: a primitive unseen in training in complex context
    # 3.1: turn left
    # 3.2: jum  p
    # 3.3.1 - 3.3.32: jump with # additional examples
    
    E_1_1 = auto() 

    E_1_2_1 = auto() 
    E_1_2_2 = auto()
    E_1_2_4 = auto()
    E_1_2_8 = auto()
    E_1_2_16 = auto()
    E_1_2_32 = auto()
    E_1_2_64 = auto()

    E_2 = auto()

    E_3_1 = auto() 
    E_3_2 = auto() 
    

    E_3_3_1_1 =auto()
    E_3_3_1_2 =auto()
    E_3_3_1_3 =auto()
    E_3_3_1_4 =auto()
    E_3_3_1_5 =auto()

    E_3_3_2_1 =auto()
    E_3_3_2_2 =auto()
    E_3_3_2_3 =auto()
    E_3_3_2_4 =auto()
    E_3_3_2_5 =auto()
    
    E_3_3_4_1 = auto() 
    E_3_3_4_2 = auto()
    E_3_3_4_3 = auto()
    E_3_3_4_4 = auto()
    E_3_3_4_5 = auto()
    
    E_3_3_8_1 = auto()
    E_3_3_8_2 = auto()
    E_3_3_8_3 = auto()
    E_3_3_8_4 = auto()
    E_3_3_8_5 = auto()
    
    E_3_3_16_1 = auto()
    E_3_3_16_2 = auto()
    E_3_3_16_3 = auto()
    E_3_3_16_4 = auto()
    E_3_3_16_5 = auto()

    E_3_3_32_1 = auto()
    E_3_3_32_2 = auto()
    E_3_3_32_3 = auto()
    E_3_3_32_4 = auto()
    E_3_3_32_5 = auto()

    MIX = auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


    def _E1_paths(self):
        E_1 = os.path.join(CURR_DIR, "dataset/data/simple_split")
        E_1_2 = os.path.join(E_1, "size_variations")
        E_1_2_train_name = lambda i: E_1_2 + f"/tasks_train_simple_p{i}.txt" 
        E_1_2_test_name = lambda i: E_1_2 + f"/tasks_test_simple_p{i}.txt" 
        if self == ExperimentType.E_1_1:
            return {"train": os.path.join(E_1, "tasks_train_simple.txt"),
                    "test": os.path.join(E_1, "tasks_test_simple.txt")}
        elif self == ExperimentType.E_1_2_1:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(1)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(1))}
        elif self == ExperimentType.E_1_2_2:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(2)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(2))}
        elif self == ExperimentType.E_1_2_4:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(4)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(4))}
        elif self == ExperimentType.E_1_2_8:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(8)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(8))}
        elif self == ExperimentType.E_1_2_16:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(16)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(16))}
        elif self == ExperimentType.E_1_2_32:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(32)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(32))}
        elif self == ExperimentType.E_1_2_64:
            return {"train": os.path.join(E_1_2, E_1_2_train_name(64)),
                    "test": os.path.join(E_1_2, E_1_2_test_name(64))}

    def _E2_paths(self):
        if self == ExperimentType.E_2:
            return {"train": os.path.join(CURR_DIR, "dataset/data/length_split/tasks_train_length.txt"),
                    "test": os.path.join(CURR_DIR, "dataset/data/length_split/tasks_test_length.txt")}

    def _E3_paths(self):
        E_3_base = os.path.join(CURR_DIR, "dataset/data/add_prim_split")
        if self == ExperimentType.E_3_1:
            return {"train": os.path.join(E_3_base, "tasks_train_addprim_turn_left.txt"),
                    "test": os.path.join(E_3_base, "tasks_test_addprim_turn_left.txt")}
        elif self == ExperimentType.E_3_2:
            return {"train": os.path.join(E_3_base, "tasks_train_addprim_jump.txt"),
                    "test": os.path.join(E_3_base, "tasks_test_addprim_jump.txt")}
        else:
            E_3_3_base = os.path.join(E_3_base, "with_additional_examples")
            base = os.path.join(E_3_base, E_3_3_base)
            sub_grps = self.name.split("_")
            add_expl = sub_grps[-2]
            rep = sub_grps[-1]

            train_tail = "tasks_train_addprim_complex_jump_num" + add_expl + "_rep"+rep+".txt"
            test_tail = "tasks_test_addprim_complex_jump_num" + add_expl + "_rep"+rep+".txt"
            return {"train": os.path.join(base, train_tail),
                    "test": os.path.join(base, test_tail)}


    def get_data_paths(self):
        if self.name.startswith("E_1"):
            return self._E1_paths()
        elif self.name.startswith("E_2"):
            return self._E2_paths()
        elif self.name.startswith("E_3"):
            return self._E3_paths()
        else:
            raise ValueError("Invalid experiment type")

from experiments.experiment_config import E1_Config, E2_Config, E3_Config


def get_config(e:ExperimentType):
    if e.name.startswith("E_1"):
        return E1_Config()
    elif e.name.startswith("E_2"):
        return E2_Config()
    elif e.name.startswith("E_3"):
        return E3_Config()
    else:
        raise ValueError("Invalid experiment type")


def get_experiments_in_group(group:str):
        """
        Retrieves a list of experiment types based on the specified group identifier.
        Args:
            group (str): The group identifier for which to retrieve experiment types. 
                    Possible values include ""(all), "1", "2", "3", "1_1", "1_2", "3_1", "3_2", "3_3" etc.
        Returns:
            list: A list of experiment types corresponding to the specified group identifier.
        """
        all = [e for e in ExperimentType]
        if group == "":
            return all
        else:
            return [e for e in all if e.name.startswith(f"E_{group}")]
        

def get_common_experiment_group(exps: list[ExperimentType]) -> str:
    # get the common group of experiments
    # e.g. E_1_1 and E_1_2_1 have common group E_1
    if len(exps) == 0:
        return ""
    group = exps[0].name
    for e in exps:
        group = os.path.commonprefix([group, e.name])
    return group
     
    

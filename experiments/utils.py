import re
import json
from experiments.evaluation_result_container import EvaluationResultContainer
import os

def is_in_ex_group(ex_group:str, ex_name:str):
    patt = r'[^0-9]+'+ex_group
    match = re.search(patt, ex_name)
    if match and match.group(0)[0] == ex_name[0]:
        return True
    return False
    
def merge_results_by_group(ex_group:str, sdir:str, tdir:str):
    # get files in dir    
    files = os.listdir(sdir)
    containers = []

    for f_name in files:
        if not is_in_ex_group(ex_group, f_name): continue
        path = os.path.join(sdir, f_name)
        print(f"Loading {f_name}")
        with open(path, "r") as f:
            container = json.load(f)
            container = EvaluationResultContainer.from_dict(container)
            containers.append(container)

    merged_containers = EvaluationResultContainer(containers)
    merged_d = EvaluationResultContainer.to_dict(merged_containers)
    #print(merged_containers)
    
    merged_name = "E"+ex_group+".json"
    path = os.path.join(tdir, merged_name)
    with open(path, "w") as f:
        json.dump(merged_d,f, indent=4)
    print(f"Merged and Saved to {path}")
    
    return merged_containers

def get_rep_numb(name:str):
    # split by '_rep_'
    name = name.split(".")[0]
    base_name, rep = name.split("_rep_")
    return base_name, int(rep)

def merge_results_by_reps(sdir:str, tdir:str):
    files = os.listdir(sdir)
    containers = {}
    rep_numbs = {}
    for f_name in files:
        path = os.path.join(sdir, f_name)
        print(f"Loading {f_name}")
        with open(path, "r") as f:
            container = json.load(f)
            container = EvaluationResultContainer.from_dict(container)
            base_name, rep = get_rep_numb(f_name)
            if base_name not in containers:
                containers[base_name] = container
                rep_numbs[base_name] = [rep]
            else:
                containers[base_name].merge_containers(container)
                rep_numbs[base_name].append(rep)

    for base_name, container in containers.items():
        reps = rep_numbs[base_name]
        reps.sort()
        reps = [str(r) for r in reps]
        reps = "+".join(reps)
        merged_d = EvaluationResultContainer.to_dict(container)
        merged_name = base_name+'_reps_'+reps+".json"
        path = os.path.join(tdir, merged_name)
        with open(path, "w") as f:
            json.dump(merged_d,f, indent=4)
        print(f"Merged and Saved to {path}")

    return containers

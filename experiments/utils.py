import re
import json
from experiments.evaluation_result_container import EvaluationResultContainer

def is_in_ex_group(ex_group:str, ex_name:str):
    patt = r'[^0-9]+'+ex_group
    match = re.search(patt, ex_name)
    if match and match.group(0)[0] == ex_name[0]:
        return True
    return False
    
def merge_results_by_group(ex_group:str, sdir:str, tdir:str):
    # get files in dir    
    import os
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



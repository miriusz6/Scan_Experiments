from difflib import SequenceMatcher

def get_common_prefix( lst: list[str]) -> str:
    if not lst:
        return ""
    if len(lst) == 1:
        return lst[0]
    lst.sort()
    s1 = lst[0]
    s2 = lst[-1]
    n = min(len(s1), len(s2))
    i = 0
    while i < n and s1[i] == s2[i]:
        i += 1
    return s1[:i]


var_fillers = list(range(ord('Z'), ord('A') - 1, -1))
var_fillers = [chr(f) for f in var_fillers]

def remove_vars(s:str):
    varsS = get_curr_fillers(s)
    return s[:-len(varsS)-2]


def remove_mods(s:str):
    modsS = get_mods(s)
    if modsS == "":
        #print("No mods")
        return s
    mods = modsS.split(",")

    varsS = get_curr_fillers(s)
    if varsS == "":
        #print("No vars")
        return s[len(modsS)+2:]

    vars = varsS.split(",")

    #print(f"Mods: {mods}")
    #print(f"Vars: {vars}")

    for mod in mods:
        if mod in vars:
            vars.remove(mod)
    s = remove_vars(s)
    return s[len(modsS)+2:] + "["+",".join(vars)+"]"
    


def get_curr_fillers(s):
    fls_st = s.find('[')
    fls_end = s.find(']')
    if fls_st == -1 or fls_end == -1:
        return ""
    #fls = s[fls_st+1:fls_end].split(',')
    fls = s[fls_st+1:fls_end]
    return fls

def get_mods(s):
    mods_st = s.find('(')
    mods_end = s.find(')')
    if mods_st == -1 or mods_end == -1:
        return ""
    return s[mods_st+1:mods_end]

def cut_fillers(s):
    fls_st = s.find('[')
    if fls_st == -1:
        return s
    return s[:fls_st]

def merge_names(s1,s2):
    if s1 == s2:
        return s1
    s_matcher = SequenceMatcher(None, s1, s2)
    s1_fls = get_curr_fillers(s1)
    s2_fls = get_curr_fillers(s2)
    s_fls_matcher = SequenceMatcher(None, s1_fls, s2_fls)
    
    s_matches = s_matcher.get_matching_blocks()
    s_fls_matches = s_fls_matcher.get_matching_blocks()
    fls_in_use = get_curr_fillers(s1).split(',') + get_curr_fillers(s2).split(',')
    avaiable_fls = [f for f in var_fillers if f not in fls_in_use]
    common = ""
    new_fls = []


    

    common_sub_strs = [s1[block.a : block.a + block.size] for block in s_matches]
    in_mods = '(' in common_sub_strs[0]
    
    for i,s in enumerate(common_sub_strs[:-1]):
        common += s
        if ')' in common:
            in_mods = False   
        if i != len(s_matches)-2 and not common_sub_strs[i+1][0] == ']':
            common += avaiable_fls[i]
            new_fls.append(avaiable_fls[i])
            if in_mods and common_sub_strs[i+1][0] != ')':
                #pew = common[i+1][0]
                common += ','
            
    if i == 0:
        return "MIXED"
    
    final_fls = []

    if len(s_fls_matches) == 1 and s_fls_matches[0].size == 0:
        pass
    else:
        for i,block in enumerate(s_fls_matches):
            if block.size == 0:
                continue
            final_fls.append(s1_fls[block.a : block.a + block.size])
    
    common = cut_fillers(common)
    final_fls += new_fls
    final_fls = ','.join(final_fls)

    return common+ f"[{final_fls}]" #f"[{fs[:-1]}]"

def merge_names2(s1,s2):
    if s1 == s2:
        return s1
    s = SequenceMatcher(None, s1, s2)
    blocks = s.get_matching_blocks()
    fillers_in_use = get_curr_fillers(s1) + get_curr_fillers(s2)
    fillers = [f for f in var_fillers if f not in fillers_in_use]
    common = ""
    new_fls = []
    
    for i,block in enumerate(blocks):
        if block.size == 0:
            continue
        common += s1[block.a : block.a + block.size]
        if i != len(blocks)-1:
            common += fillers[i]
            new_fls.append(fillers[i])
    if i == 0:
        return "MIXED"
    

    prev_fs_i = common.find('[')
    if prev_fs_i != -1:
        common = common[0:prev_fs_i]

    return common+ f"[{new_fls}]" #f"[{fs[:-1]}]"


from collections import OrderedDict


def flatten_dict(d, parent_key=None):
    items = []
    for k, v in d.items():
        #new_key = parent_key + sep + k if parent_key else k
        if parent_key == None:
            new_key = (k,)
        else:
            new_key = tuple( list(parent_key) + [k] )

        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)

def unflatten_dict(d):
    items = {}
    for k, v in d.items():
        keys = list(k)
        sub_items = items
        for key in keys[:-1]:
            sub_items = sub_items.setdefault(key, {})
        sub_items[keys[-1]] = v
    return items



def switch_dict_lvls(d, new_indxs):
    flat = flatten_dict(d)
    new_flat = OrderedDict()
    for k,v in flat.items():
        ks = list(k)
        ks = [ks[i] for i in new_indxs]
        ks = tuple(ks)
        new_flat[ks] = v
    return unflatten_dict(new_flat)


def nice_dict_print(d, indent=0):
    for k,v in d.items():
        if isinstance(v, dict):
            print(" "*indent + k)
            nice_dict_print(v, indent+2)
        else:
            print(" "*indent + k + " : " + str(v))

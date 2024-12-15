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

def merge_names(s1,s2):
    if s1 == s2:
        return s1
    s = SequenceMatcher(None, s1, s2)
    blocks = s.get_matching_blocks()
    fillers = list(range(ord('Z'), ord('A') - 1, -1))
    fillers = [chr(f) for f in fillers]
    common = ""
    fs = ""
    for i in range(len(blocks)):
        common += s1[blocks[i].a : blocks[i].a + blocks[i].size] + fillers[i]
        fs += fillers[i]
    if i == 0:
        return "MIXED"
    prev_fs_i = common.find('[')
    if prev_fs_i != -1:
        common = common[0:prev_fs_i]

    return common[0:-1]+ f"[{fs[:-1]}]"


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

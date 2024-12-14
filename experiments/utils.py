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

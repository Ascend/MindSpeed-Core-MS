import re
num_npu = 8


def parse_memory_file(fname):
    p_memory = r'\| \d.*\| \d+.*\|.*\| (\d+).*\|'
    with open(fname, 'r') as f:
        context = f.read().split('\n')
    try:
        mems = []
        for l in context:
            m = re.match(p_memory, l)
            if m:
                mems.append(int(m.group(1)))
        max_mem = max(mems)
    except:
        max_mem = None
    return max_mem / 1024


def parse_script(file):
    with open(file, 'r') as f:
        context = f.read().split('\n')
    p_gbs = r'.*global-batch-size (\d*).*'
    p_len = r'.*seq-length (\d*).*'
    gbs, len = None, None
    for l in context:
        match = re.match(p_gbs, l)
        if match:
            gbs = match.group(1)
        match = re.match(p_len, l)
        if match:
            len = match.group(1)
    return gbs, len


def parse_log_file(file):
    it_pattern = (r'.*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] '
                  r'iteration\s*(\d*)\/.*lm loss: ([\d\.]*).*grad norm: ([\d\.]*).*')
    with open(file, 'r') as f:
        context = f.read().split('\n')
    data = {}
    for l in context:
        match = re.match(it_pattern, l)
        if match:
            data[int(match.group(2))] = match.groups()
    return data

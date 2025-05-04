# pip install anytree
import pandas as pd
from collections import defaultdict
from anytree import Node, RenderTree

# Load and preprocess data
df = pd.read_csv("/content/Association Rule Mining.csv")
t = df['Items'].dropna().apply(lambda x: x.strip().upper().split())
n = len(t)

# Minimum support input
ms = round(float(input('Min support %: ')) / 100 * n)
print(f'\nMin Supp: {ms}')

# Count item frequencies
ic = defaultdict(int)
for x in t:
    for i in x:
        ic[i] += 1

# Filter by min support
ic = {k: v for k, v in ic.items() if v >= ms}

# Sort and filter transactions
def si(x): return sorted([i for i in x if i in ic], key=lambda y: (-ic[y], y))
ot = [si(x) for x in t if si(x)]

# Define FP-Tree node
class N:
    def __init__(self, i, c, p):
        self.i, self.c, self.p = i, c, p
        self.ch, self.l = {}, None

# Build FP-Tree
def bft(tr):
    r = N(None, 0, None)
    h = defaultdict(list)
    for x in tr:
        c = r
        for i in x:
            if i in c.ch:
                c.ch[i].c += 1
            else:
                c.ch[i] = N(i, 1, c)
                h[i].append(c.ch[i])
            c = c.ch[i]
    return r

# Traverse FP-Tree with anytree
def ba(n, p=None):
    m = f"{n.i} ({n.c})" if n.i else ";Root"
    x = Node(m, p)
    for c in n.ch.values():
        ba(c, x)
    return x

# Build the FP-Tree
r = bft(ot)

# Print FP-Tree Structure
print('\nFP-Tree Structure:')
tr = ba(r)
for pre, _, node in RenderTree(tr):
    print(f"{pre}{node.name}")

# -----------------------------------------------
# Conditional Pattern Base Table (formatted clean)
# -----------------------------------------------
header_table = defaultdict(list)

def link_header(n):
    if n.i:
        header_table[n.i].append(n)
    for child in n.ch.values():
        link_header(child)

link_header(r)

def get_conditional_pattern_base(item):
    paths = []
    for node in header_table[item]:
        path = []
        p = node.p
        while p and p.i is not None:
            path.append(p.i)
            p = p.p
        if path:
            paths.append((list(reversed(path)), node.c))
    return paths

def build_conditional_fp_tree(pattern_base):
    tree = defaultdict(int)
    for path, count in pattern_base:
        for item in path:
            tree[item] += count
    return dict(tree)

def generate_patterns(item, conditional_fp_tree):
    return [tuple(sorted((item,) + (i,))) for i in conditional_fp_tree]

# Build final DataFrame
rows = []
for item in sorted(header_table.keys(), key=lambda x: ic[x]):
    cpb = get_conditional_pattern_base(item)
    cpb_str = [f"({' -> '.join(p)}):{c}" for p, c in cpb]
    cond_fp_tree = build_conditional_fp_tree(cpb)
    freq_patterns = generate_patterns(item, cond_fp_tree)

    rows.append({
        'Item': item,
        'Conditional Pattern Base': ', '.join(cpb_str) if cpb_str else '∅',
        'Conditional FP-Tree': ', '.join([f"{k}:{v}" for k, v in cond_fp_tree.items()]) if cond_fp_tree else '∅',
        'Frequent Pattern Generation': ', '.join(['{' + ','.join(p) + '}' for p in freq_patterns]) if freq_patterns else '∅'
    })

cond_df = pd.DataFrame(rows)

# Print Final Table
print("\nConditional Pattern Base Table:\n")
print(cond_df.to_string(index=False))


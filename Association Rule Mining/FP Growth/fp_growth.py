from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Load dataset
data = pd.read_csv('/content/Association Rule Mining.csv')
transactions = data['Items'].dropna().apply(lambda x: x.strip().upper().split()).tolist()

# User input for support
min_support = float(input("Minimum support % : "))
min_support_count = int((min_support / 100) * len(transactions))

# Tree Node class
class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

    def increment(self, count):
        self.count += count

    def display(self, ind=1):
        print('  ' * ind, self.item, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)

# Insert into the FP-tree
def insert_tree(items, node, header_table):
    if not items:
        return
    first = items[0]
    if first in node.children:
        node.children[first].increment(1)
    else:
        node.children[first] = TreeNode(first, 1, node)
        header_table[first].append(node.children[first])
    insert_tree(items[1:], node.children[first], header_table)

# Build the FP-tree
def build_fp_tree(transactions, min_support_count):
    item_count = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_count[item] += 1
    item_count = {item: count for item, count in item_count.items() if count >= min_support_count}
    freq_items = sorted(item_count.items(), key=lambda x: (-x[1], x[0]))

    print("\nInitial FP-Table (after support filtering):")
    for item, count in freq_items:
        print(f"Item: {item}, Count: {count}")

    item_order = {item: idx for idx, (item, _) in enumerate(freq_items)}
    root = TreeNode(None, 1, None)
    header_table = defaultdict(list)

    for tidx, transaction in enumerate(transactions):
        ordered_items = [item for item in sorted(transaction, key=lambda x: item_order.get(x, float('inf'))) if item in item_order]
        print(f"\nTransaction {tidx + 1}: {ordered_items}")
        insert_tree(ordered_items, root, header_table)

        print("\nFP-Tree after inserting this transaction:")
        root.display()

    return root, header_table, item_count

# Get conditional pattern base
def find_prefix_paths(base_item, nodes):
    conditional_patterns = []
    for node in nodes:
        path = []
        parent = node.parent
        while parent and parent.item:
            path.append(parent.item)
            parent = parent.parent
        if path:
            conditional_patterns.append((path[::-1], node.count))
    return conditional_patterns

# Build conditional tree item count
def build_conditional_tree(conditional_patterns, min_support_count):
    cond_transactions = []
    for path, count in conditional_patterns:
        for _ in range(count):
            cond_transactions.append(path)
    cond_item_count = defaultdict(int)
    for trans in cond_transactions:
        for item in trans:
            cond_item_count[item] += 1
    cond_item_count = {item: count for item, count in cond_item_count.items() if count >= min_support_count}
    return cond_item_count

# Generate frequent patterns
def generate_patterns(base_item, cond_tree):
    patterns = []
    for item in cond_tree:
        pattern = [base_item, item]
        patterns.append("-".join(sorted(pattern)))
    if len(cond_tree) > 1:
        for item1 in cond_tree:
            for item2 in cond_tree:
                if item1 < item2:
                    patterns.append("-".join(sorted([base_item, item1, item2])))
    return patterns

# Run FP-Growth
fp_root, header_table, item_supports = build_fp_tree(transactions, min_support_count)

print("\nFinal FP-Tree Structure:")
fp_root.display()

# Simulate FP-Growth process
table_data = []
for item in sorted(header_table.keys()):
    cpb_raw = find_prefix_paths(item, header_table[item])
    cpb = [set(p[0]) for p in cpb_raw]
    cond_tree = build_conditional_tree(cpb_raw, min_support_count)
    patterns = generate_patterns(item, cond_tree)
    table_data.append([
        item,
        str(cpb),
        str(cond_tree),
        str(patterns)
    ])

print("\nSimulated FP-Growth Table:")
print(tabulate(table_data, headers=["Item", "Conditional Pattern Base", "Conditional FP-Tree", "Frequent Patterns"], tablefmt="grid"))


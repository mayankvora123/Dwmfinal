# Input Format
# I1,I2,I5
# I2,I4
# I2,I3
# I1,I2,I4
# I1,I3
# I2,I3
# I1,I3
# I1,I2,I3,I5
# I1,I2,I3

# 22
# 60

import pandas as pd
from itertools import combinations

# --- Load Excel File ---
# df = pd.read_csv('/content/Association Rule Mining.csv')

# --- User Input ---
def get_transactions_from_user():
    while True:
        try:
            n = int(input("Enter the number of transactions: ").strip())
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    transactions = []
    print("\nEnter each transaction (items separated by commas, e.g., I1,I2,I5):")

    for i in range(n):
        while True:
            user_input = input(f"Transaction {i + 1}: ").strip()
            if user_input:
                transaction = [item.strip() for item in user_input.split(',') if item.strip()]
                transactions.append(transaction)
                break
            else:
                print("Transaction cannot be empty. Please enter again.")

    return transactions

# Example usage
transactions = get_transactions_from_user()

print("\nFinal Transactions List:")
for i, t in enumerate(transactions, 1):
    print(f"T{i}: {t}")

print()

# --- Sample Transactions ---
# transactions = [
#     ['I1', 'I2', 'I5'],
#     ['I2', 'I4'],
#     ['I2', 'I3'],
#     ['I1', 'I2', 'I4'],
#     ['I1', 'I3'],
#     ['I2', 'I3'],
#     ['I1', 'I3'],
#     ['I1', 'I2', 'I3', 'I5'],
#     ['I1', 'I2', 'I3']
# ]

# --- Preprocess Transactions ---
# transactions = df['List of item Ids'].dropna().apply(lambda x: set(str(x).strip().lower().split(' '))).tolist()
num_transactions = len(transactions)

# --- User Input ---
min_support_percent = float(input("Enter minimum support %: "))
min_confidence_percent = float(input("Enter minimum confidence %: "))

# --- Step 1: Minimum Support Count ---
min_support_count = round((min_support_percent * num_transactions) / 100)
print(f"\nMinimum Support Count: {min_support_count}")

# --- Support Count Helper ---
def count_support(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(t))

# --- Step 2: Generate C1 and L1 ---
item_counts = {}
for transaction in transactions:
    for item in transaction:
        itemset = frozenset([item])
        item_counts[itemset] = item_counts.get(itemset, 0) + 1

C1 = item_counts
L1 = {itemset: count for itemset, count in C1.items() if count >= min_support_count}
frequent_itemsets = [L1]
k = 2

# --- Step 3: Generate Frequent Itemsets ---
while True:
    prev_Lk = list(frequent_itemsets[-1].keys())
    candidates = []

    for i in range(len(prev_Lk)):
        for j in range(i + 1, len(prev_Lk)):
            union = prev_Lk[i].union(prev_Lk[j])
            if len(union) == k and union not in candidates:
                subsets = list(combinations(union, k - 1))
                if all(frozenset(s) in prev_Lk for s in subsets):
                    candidates.append(union)

    Ck = {}
    for candidate in candidates:
        count = count_support(candidate, transactions)
        if count >= min_support_count:
            Ck[frozenset(candidate)] = count

    if not Ck:
        break

    frequent_itemsets.append(Ck)
    k += 1

# --- Step 4: Show Final Frequent Itemsets ---
final_frequent = frequent_itemsets[-1]
print("\n Final Frequent Itemsets:")
for itemset in final_frequent:
    print(set(itemset))

# --- Step 5: All Association Rules + Strong Rules ---
all_rules = []
strong_rules = []

print("\n All Association Rules with Confidence:")
for itemset, support_count in final_frequent.items():
    for i in range(1, len(itemset)):
        for lhs in combinations(itemset, i):
            lhs = frozenset(lhs)
            rhs = itemset - lhs
            if rhs:
                lhs_count = count_support(lhs, transactions)
                if lhs_count > 0:
                    confidence = (support_count / lhs_count) * 100
                    confidence = round(confidence, 2)
                    print(f"{set(lhs)} => {set(rhs)} | confidence = {confidence}%")
                    all_rules.append((lhs, rhs, confidence))
                    if confidence >= min_confidence_percent:
                        strong_rules.append((lhs, rhs, confidence))

# --- Step 6: Strong Association Rules ---
print("\n Strong Association Rules (confidence ≥ " + str(min_confidence_percent) + "%):")
for lhs, rhs, confidence in strong_rules:
    print(f"{set(lhs)} => {set(rhs)} | confidence = {confidence}%")


import sys
from collections import defaultdict
from itertools import combinations

# Load transactions from file
def load_transactions(input_file):
    transactions = []
    with open(input_file, 'r') as f:
        for line in f:
            transaction = set(map(int, line.strip().split('\t')))
            transactions.append(transaction)
    return transactions

# Compute frequent 1-itemsets
def frequent_items(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    n = len(transactions)
    min_count = n * min_support / 100
    frequent_items = set(frozenset([item]) for item, count in item_counts.items() if count >= min_count)

    return frequent_items

def apriori(transactions, min_support):
    itemsets = []
    # k is the number of items in the itemset
    k = 1
    # Compute frequent 1-itemsets
    frequent = frequent_items(transactions, min_support)
    itemsets.append(frequent)

    # Compute frequent k-itemsets
    # first calculate
    while len(itemsets[k-1]) > 0:
        candidates = set()
        for itemset in itemsets[k-1]:
            for item in frequent:
                candidate = itemset | item
                if len(candidate) == k+1:
                    candidates.add(candidate)

        counts = defaultdict(int)
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[candidate] += 1

        n = len(transactions)
        min_count = n * min_support / 100
        frequent = set(candidate for candidate, count in counts.items() if count >= min_count)
        itemsets.append(frequent)
        k += 1

    return itemsets[:-1]

# then find the results that meet the min_confidence
def association_rules(itemsets, min_support, min_confidence):
    transaction_length = len(transactions)
    rules = []
    # iterate through the itemsets
    for i in range(1, len(itemsets)):
        # iterate through the itemsets in the current itemset
        for itemset in itemsets[i]:
            # iterate through the number of items in the current itemset
            for j in range(1, i+1):
                # iterate through the antecedent
                for antecedent in combinations(itemset, j):
                    consequent = itemset - set(antecedent)
                    support = sum(1 for transaction in transactions if itemset.issubset(transaction)) / transaction_length  * 100
                    if support < min_support:
                        continue

                    antecedent_set = set(antecedent)
                    confidence = sum(1 for transaction in transactions if itemset.issubset(transaction) and antecedent_set.issubset(transaction)) \
                                 / sum(1 for transaction in transactions if antecedent_set.issubset(transaction)) * 100
                    if confidence >= min_confidence:
                        rule = {'antecedent': frozenset(antecedent), 'consequent': frozenset(consequent),
                                'support': round(support, 2), 'confidence': round(confidence, 2)}
                        rules.append(rule)

    return rules

if __name__ == '__main__':
    min_support = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    transactions = load_transactions(input_file)
    itemsets = apriori(transactions, min_support)
    rules = association_rules(itemsets, min_support, 0)

    with open(output_file, 'w') as f:
        for rule in rules:
            f.write('{{{}}}\t{{{}}}\t{:.2f}\t{:.2f}\n'.format(
                ','.join(map(str, rule['antecedent'])), ','.join(map(str, rule['consequent'])),
                rule['support'], rule['confidence']))
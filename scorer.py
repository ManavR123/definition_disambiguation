from collections import defaultdict


def score_expansion(key, prediction):
    correct = 0
    for i in range(len(key)):
        if key[i] == prediction[i]:
            correct += 1
    acc = correct / len(prediction)

    expansions = set()
    correct_per_expansion = defaultdict(int)
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)
    for i in range(len(key)):
        expansions.add(key[i])
        total_per_expansion[key[i]] += 1
        pred_per_expansion[prediction[i]] += 1
        if key[i] == prediction[i]:
            correct_per_expansion[key[i]] += 1

    precs = defaultdict(int)
    recalls = defaultdict(int)

    for exp in expansions:
        precs[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = 2 * macro_prec * macro_recall / (macro_prec + macro_recall) if macro_prec + macro_recall != 0 else 0

    return acc, macro_prec, macro_recall, macro_f1
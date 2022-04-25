import json

import matplotlib.pyplot as plt
import numpy as np


terms = json.load(open("wikipedia_parsing/ambiguous_term_to_senses.json"))
data = [len(term) for term in terms.values()]
arr = plt.hist(data, bins=np.arange(2, 9))
plt.ylabel("Frequency")
plt.xlabel("Number of Senses")
plt.xticks(np.arange(2, 9))
plt.title(f"Sense Frequencies for Ambiguous Wikipedia Terms (n={len(terms)})")
for i in range(0, 6):
    plt.text(arr[1][i] + 0.25, arr[0][i], str(int(arr[0][i])))

plt.savefig("wikipedia_parsing/ambiguous_terms_sense_frequency_distribution.png")

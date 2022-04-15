import json

import matplotlib.pyplot as plt
import numpy as np


terms = json.load(open("wikipedia_parsing/ambiguous_term_to_senses.json"))
data = [len(term) for term in terms]
plt.hist(data, bins=np.arange(1, max(data) + 2))
plt.ylabel("Frequency")
plt.xlabel("Number of Senses")
plt.xticks(np.arange(1, max(data) + 2))
plt.title(f"Sense Frequencies for Ambiguous Wikipedia Terms (n={len(terms)})")
plt.savefig("wikipedia_parsing/ambiguous_terms_sense_frequency_distribution.png")

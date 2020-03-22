import random
import numpy as np


for _ in range(10):
    a = np.arange(10)
    random.seed(34)
    print(random.choice(a))

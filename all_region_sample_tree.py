import numpy as np
import random



def all_region_sample(lower_bound, upper_bound, num = 100, seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    interval = np.array([lower_bound, upper_bound]).T
    new_data = []
    for j in range(int(num)):
        new_data.append([random.uniform(interval[i, 0], interval[i, 1]) for i in range(interval.shape[0])])
    new_data = np.array(new_data)
    return new_data
from collections import deque
import numpy as np
import itertools


class Clustering:

    def __init__(self):
        self.history = {
            'd_max': deque(), 'd_min': deque(), 'sigma_d': deque(),
            'mu_d': deque(), 'V': deque(), 'E[tr]': deque(), 'D': deque()
        }

    def calculate_indices(self, locations):
        c = len(locations)
        clockwise_distance_D = 0.0
        if c == 2:
            # D_n = (location_2 - location_1) mod 1
            clockwise_distance_D = (locations[1] - locations[0]) % 1.0
        sorted_locs = np.sort(locations)

        distances = np.diff(sorted_locs)
        wrap_around_dist = (1.0 - sorted_locs[-1]) + sorted_locs[0]
        distances = np.append(distances, wrap_around_dist)

        # Calculate D_max
        d_max = np.max(distances)
        d_max = np.round(d_max, decimals=15)
        if d_max < 1 / c:
            d_max = 1.0

        # Calculate D_min
        d_min = np.min(distances)
        d_min = np.round(d_min, decimals=15)
        if d_min > 1 / c:
            d_min = 0.0

        # Calculate E[tr]
        e_tr = np.sum(distances ** 2)
        if e_tr == 0.0:
            e_tr = 0.5
        else:
            e_tr = e_tr / 2
        e_tr = np.round(e_tr, decimals=15)

        metrics = {'d_max': d_max, 'd_min': d_min, 'E[tr]': e_tr, 'D': clockwise_distance_D}

        for key, value in metrics.items():
            self.history[key].append(value)

        return metrics
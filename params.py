import numpy as np

head_conv = 256
num_layers = 34
down_ratio = 4
scale = 1.0
list_label = ['p']              # only edit here
num_classes = len(list_label)
max_per_image = 20
heads = {'hm': num_classes, 'reg': 2}

mean = np.array([0.472459, 0.475080, 0.482652],
                dtype=np.float32).reshape((1, 1, 3))
std = np.array([0.255084, 0.254665, 0.257073],
               dtype=np.float32).reshape((1, 1, 3))
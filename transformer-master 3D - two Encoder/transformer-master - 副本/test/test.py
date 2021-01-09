import numpy as np

a = np.loadtxt('test.txt', skiprows=1, dtype=int, comments='#',usecols=(0, 2), unpack=True)
print(a)
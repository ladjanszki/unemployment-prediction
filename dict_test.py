import numpy as np

I = np.array([1, 1, 1, 1])
a = np.array([1, 1, 0, 0])
b = np.array([0, 0, 0, 1])

asc = 0.5
bsc = 0.2


#ainv = np.subtract(I, a)
ainv = I-a
binv = I-b
#binv = np.subtract(I, b)

print(ainv)
print(binv)
print(np.add(ainv, binv))
print(asc * ainv + bsc * binv)



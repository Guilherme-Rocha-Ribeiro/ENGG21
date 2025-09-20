import numpy as np
import timeit

# Create some sample data (similar to your use case)
x_train = np.random.rand(1000, 1)  # 1000 samples, 1 feature
params = np.random.rand(1, 1)      # Parameter matrix

# Test np.dot()
def test_dot():
    return np.dot(x_train, params)

# Test @ operator  
def test_at():
    return x_train @ params

def test_matmul():
    return np.matmul(x_train, params)
# Time both functions

time_dot = []
time_at = []
time_matmult = []
for i in range(100):
    dot_time = timeit.timeit(test_dot, number=10000)
    at_time = timeit.timeit(test_at, number=10000)
    matmul_time = timeit.timeit(test_matmul, number=10000)

    time_at.append(at_time)
    time_dot.append(dot_time)
    time_matmult.append(matmul_time)

mean_dot = sum(time_dot) / len(time_dot)
mean_at = sum(time_at) / len(time_at)
mean_matmul = sum(time_matmult) / len(time_matmult)

print(mean_dot)
print(mean_at)
print(mean_matmul)
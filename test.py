import math

iter_num = 10000

ratio = 1.25 ** math.log(iter_num + 1)

print(ratio)

train_size = 1024

print(train_size * ratio)  # 2




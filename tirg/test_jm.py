import torch
import numpy as np

x = torch.rand(5,3)
y = np.array([[1,2],[3,3]])

print('x: ',x)
print('cpu:', torch.cuda.is_available())
print('y: ', y)
print(x.numpy())

# print('version: ', torch.__version__)
# print(torch.backends.mps.is_available())  # Should print True

# import os

# # Correct the path
# path = './fashion200k/women/dresses/casual_and_day_dresses/51727804/51727804_0.jpeg'

# # Check if the path exists
# if os.path.exists(path):
#     print(f"The file exists at {path}")
# else:
#     print(f"File not found: {path}")

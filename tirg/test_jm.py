import torch
import numpy as np

nn_result = []
with open('nn_results.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # 读取所有行到列表中
    for line in lines:
        shenhai = line.strip().split()  # 去除首尾空格并按空格分割
        nn_result.append(shenhai)  # 将每行的结果添加到nn_result列表中
file.close()
print('nn_result:\n',type(nn_result))
print('nn_result.shape:\n','(',len(nn_result),',',len(nn_result[0]),')')

all_target_captions = []
with open('all_target_captions.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # 读取所有行到列表中
    for line in lines:
        shenhai = line.strip().replace(" ", "")  # 去除首尾空格并按空格分割
        all_target_captions.append(shenhai)  # 将每行的结果添加到nn_result列表中
file.close()
print('all_target_captions:\n',type(all_target_captions))
print('all_target_captions.shape:\n','(',len(all_target_captions),',',len(all_target_captions[0]),')')

test_queries = []
with open('../fashion200k/test_queries.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # 读取所有行到列表中
    for line in lines:
        shenhai = line.strip().split()  # 去除首尾空格并按空格分割
        test_queries.append(shenhai)  # 将每行的结果添加到nn_result列表中

k = 1
for i, nns in enumerate(nn_result):
    #   i: the i-th test query
    # nns: the sorted neighbors of the i-th query
      if all_target_captions[i] in nns[:k]: # the target caption in the top k results
        print('i:', i)

# nn_result = [['苹果','梨'],['樱桃','橘子','芒果'],['早饭','莲雾','释迦','荔枝'],['火龙果','鸡丝凉面']]
# with open("output.txt", "w") as file:
#     for row in nn_result:
#         # 将每个元素转为字符串并用空格连接，最后加换行符
#         line = " ".join(map(str, row)) + "\n"
#         file.write(line)

# f = open("nn_results.txt")
# nn_results = f.read()
# for i, nns in enumerate(nn_results):
#     # i: the i-th test query
#     # nns: the sorted neighbors of the i-th query
#     if all_target_captions[i] in nns[:k]:
#         r += 1

# x = torch.rand(5,3)
# y = np.array([[1,2],[3,3]])

# print('x: ',x)
# print('cpu:', torch.cuda.is_available())
# print('y: ', y)
# print(x.numpy())

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
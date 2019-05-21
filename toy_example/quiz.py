# nihao = [[1,2],[3,4],[5,6]]
# for i in range(3):
#     for j in range(2):
#         if i == 2 and j == 1:
#             break
#         print(nihao[i][j])
#
#     print('-----')
import numpy as np
def choosing_k(k, candidate_path, candidate_prob):
    candidate_path = np.array(candidate_path)
    # new = []
    # for i in candidate_path:
    #     new += i
    # new = np.array( new )
    # candidate_path = new
    candidate_prob = np.array(candidate_prob)
    new = np.append(candidate_path, candidate_prob, axis = 1)
    new = list(new)
    new.sort(key = lambda x:x[-1], reverse=True)
    new = new[:k]
    for i in range( len( new ) ):
        new[i] = list( new[i] )
    return new
li = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
# li = np.array(li)
# print(li)
# ni = np.array(ni)
# # for i in ni:
# #     print(i)
# hao = ni[:,-2]
#
# # print(hao)
# li = []
# for i in hao:
#     li += i
#
#
#
# li = np.array(li)
# print(li)
jia = [[1], [7], [3], [9], [8], [2], [1], [9], [9]]
# jia = np.array(jia)
# print(jia)
# #
# jia = np.array(jia).reshape(-1,1)
# print(jia)
# lun = np.append(li, jia, axis = 1)
#
#
# lun = list(lun)
# lun.sort(key = lambda x:x[-1], reverse=True)
# print(lun)
nihao = choosing_k(5, li, jia)

print(nihao)

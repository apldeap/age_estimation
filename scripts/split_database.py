import numpy as np

shuffle_1_lst = []
shuffle_2_lst = []

with open('shuffle1.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        shuffle_1_lst.append(int(line))

with open('shuffle2.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        shuffle_2_lst.append(int(line))

shuffle_1 = np.array(shuffle_1_lst)
shuffle_2 = np.array(shuffle_2_lst)

image_names = []

for i in range(81081):
    image_names.append('face_' + str(shuffle_1[i]) + '.png')

train_names = [None] * 64864
test_names = [None] * 16217

for i in range(81081):
    index = shuffle_2[i] - 1
    if index <= 64863:
        train_names[index] = image_names[i]
    else:
        index -= 64864
        test_names[index] = image_names[i]

f = open('train_names.txt', 'w')
for name in train_names:
    f.write(name + '\n')
f.close()

f = open('test_names.txt', 'w')
for name in test_names:
    f.write(name + '\n')
f.close()
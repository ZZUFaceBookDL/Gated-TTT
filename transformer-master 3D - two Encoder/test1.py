"""
    测试数据集的各个参数
"""


from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import torch.utils.data as Data
import re

# 数据集路径选择
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\ArabicDigits\\ArabicDigits.mat'  # lenth=6600  input=93 channel=13 output=10
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\AUSLAN\\AUSLAN.mat'  # lenth=1140  input=136 channel=22 output=95
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\CharacterTrajectories\\CharacterTrajectories.mat'
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\CMUsubject16\\CMUsubject16.mat'  # lenth=29,29  input=580 channel=62 output=2
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\JapaneseVowels\\JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\Libras\\Libras.mat'  # lenth=180  input=45 channel=2 output=15
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\UWave\\UWave.mat'  # lenth=4278  input=315 channel=3 output=8
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\Wafer\\Wafer.mat'  # lenth=896  input=198 channel=6 output=2  换

path = 'E:\\PyCharmWorkSpace\\mtsdata\\WalkvsRun\\WalkvsRun.mat'  # lenth=28  input=1918 channel=62 output=2  换
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\KickvsPunch\\KickvsPunch.mat'  # lenth=10  input=841 channel=62 output=2
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\NetFlow\\NetFlow.mat'  # lenth=803  input=997 channel=4 output=只有1和13
# path = 'E:\\PyCharmWorkSpace\\mtsdata\\PEMS\\PEMS.mat'  # lenth=267  input=144 channel=963 output=7


# 读数据
m = loadmat(path)
# print(len(m))  # 4
# print(m)  # dtype=[('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]

# m中是一个字典 有4个key 其中最后一个键值对存储的是数据
x1, x2, x3, x4 = m
data = m[x4]
# print('x4 ', data)
# print('data.type ', type(data))  # <class 'numpy.ndarray'>
print('x4.dtype ', data.dtype)  # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]

# index_train = re.search('train\'', str(data.dtype))
# print(index_train)

# print(str(data.dtype))
index_train = str(data.dtype).find('train\'')
index_trainlabels = str(data.dtype).find('trainlabels')
index_test = str(data.dtype).find('test\'')
index_testlabels = str(data.dtype).find('testlabels')
list = [index_test,  index_train, index_testlabels, index_trainlabels]
list = sorted(list)
index_train = list.index(index_train)
index_trainlabels = list.index(index_trainlabels)
index_test = list.index(index_test)
index_testlabels = list.index(index_testlabels)
# print(index_train, index_trainlabels, index_test, index_testlabels)

# print('x4.shape ', data.shape)  # (1, 1)

data0 = data[0]
# print('data0.shape', data0.shape)  # (1,)
data00 = data[0][0]
# print('data00.shape', data00.shape)  # ()  data00才到达数据的维度

# [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]  O 表示数据类型为 numpy.object
# 注：不同数据集顺序不同 ！！！！！！！！！！！！！！！！！！！
train_lable = data00[index_trainlabels]
train_data = data00[index_train]
test_lable = data00[index_testlabels]
test_data = data00[index_test]

print('train_lable ', train_lable.shape)  # (6600, 1)
print('train_data', train_data.shape)  # (1, 6600)
print('test_lable', test_lable.shape)  # (2200, 1)
print('test_data', test_data.shape)  # (1, 2200)

train_lable = train_lable.squeeze()
# print('train_lable ', train_lable.shape)  # (6600)
train_data = train_data.squeeze()
# print('train_data', train_data.shape)  # (6600,)
test_lable = test_lable.squeeze()
test_data = test_data.squeeze()

# 数据最大时间步数
max_lenth = 0  # 93
for item in train_data:
    item = torch.as_tensor(item).float()
    # print(item.shape)
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]

for item in test_data:
    item = torch.as_tensor(item).float()
    # print(item.shape)
    if item.shape[1] > max_lenth:
        max_lenth = item.shape[1]

print('max_lenth 最大时间步 即input', max_lenth)  # 93

train_dataset = []
test_dataset = []
for x1 in train_data:
    x1 = torch.as_tensor(x1).float()
    if x1.shape[1] != max_lenth:
        # padding填充
        padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
        x1 = torch.cat((x1, padding), dim=1)
    train_dataset.append(x1)

for x2 in test_data:
    x2 = torch.as_tensor(x2).float()
    if x2.shape[1] != max_lenth:
        # padding填充
        padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
        x2 = torch.cat((x2, padding), dim=1)
    test_dataset.append(x2)

# 得到结果为 [数据条数,时间步数,时间序列维度]
train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
train_label = train_lable
test_label = test_lable
test_label_length = len(tuple(set(train_lable)))
train_label_length = len(set(test_lable))
# print(test_lable[500])
print('训练集标签值长度', train_label_length)  # 10
print('测试集标签值长度', train_label_length)
print('[测试集数据条数, 时间步最大值, 时间序列维度]', train_dataset.shape)

# print(train_dataset.shape).permute(0, 2, 1)

print('\r\n\r\n接下来是数据集对象测试=====================================')
# 数据集对象测试
class OzeDataset(Dataset):
    def __init__(self, filepath, dataset):
        super(OzeDataset, self).__init__()
        self.dataset = dataset
        m = loadmat(filepath)

        # m中是一个字典 有4个key 其中最后一个键值对存储的是数据
        x1, x2, x3, x4 = m
        data = m[x4]
        # print('x4 ', data)
        # print('data.type ', type(data))  # <class 'numpy.ndarray'>
        print('x4.dtype ', data.dtype)  # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]
        # print('x4.shape ', data.shape)  # (1, 1)

        data0 = data[0]
        # print('data0.shape', data0.shape)  # (1,)
        data00 = data[0][0]
        # print('data00.shape', data00.shape)  # ()  data00才到达数据的维度

        index_train = str(data.dtype).find('train\'')
        index_trainlabels = str(data.dtype).find('trainlabels')
        index_test = str(data.dtype).find('test\'')
        index_testlabels = str(data.dtype).find('testlabels')
        list = [index_test, index_train, index_testlabels, index_trainlabels]
        list = sorted(list)
        index_train = list.index(index_train)
        index_trainlabels = list.index(index_trainlabels)
        index_test = list.index(index_test)
        index_testlabels = list.index(index_testlabels)


        # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]  O 表示数据类型为 numpy.object
        train_lable = data00[index_trainlabels]
        train_data = data00[index_train]
        test_lable = data00[index_testlabels]
        test_data = data00[index_test]

        train_lable = train_lable.squeeze()
        train_data = train_data.squeeze()
        test_lable = test_lable.squeeze()
        test_data = test_data.squeeze()

        self.train_len = train_data.shape[0]
        self.test_len = test_data.shape[0]

        self.output = len(tuple(set(train_lable)))

        max_lenth = 0  # 93
        for item in train_data:
            item = torch.as_tensor(item).float()
            # print(item.shape)
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        for item in test_data:
            item = torch.as_tensor(item).float()
            # print(item.shape)
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        train_dataset = []
        test_dataset = []
        for x1 in train_data:
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            # x1 = x1[:, :min_lenth]
            # print(item.shape)
            train_dataset.append(x1)

        for x2 in test_data:
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            # x2 = x2[:, :min_lenth]
            # print(item.shape)
            test_dataset.append(x2)

        self.train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
        print(self.train_dataset.shape)
        self.test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
        self.train_label = torch.Tensor(train_lable)
        self.test_label = torch.Tensor(test_lable)
        self.channel = self.test_dataset[0].shape[-1]
        self.input = self.test_dataset[0].shape[-2]

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index] - 1
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index] - 1

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len


dataset_train = OzeDataset(path, 'train')
dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=20, shuffle=False)
dataset_test = OzeDataset(path, 'test')
dataloader_test = Data.DataLoader(dataset=dataset_test, batch_size=20, shuffle=False)

print(dataset_test.train_len, dataset_test.input, dataset_test.channel,  dataset_test.output)


# 测试dataloader_train， dataloader_test 中读取的数据是否相同  相同位置为1 不同为0
list1 = []
list2 = []
for x, y in dataloader_train:
    list1.append(x)

for x, y in dataloader_test:
    list2.append(x)

print(f'self.train_len={dataset_train.train_len}\r\nself.test_len={dataset_train.test_len}')
print('[bathsize, input, channel] =', list1[0].shape)

# print('\r\n测试dataloader_train， dataloader_test 中读取的数据是否相同  相同位置为1 不同为0')
# print(torch.eq(list1[0][0], list2[0][0]))
# print(torch.eq(torch.Tensor([1, 2, 3]), torch.Tensor([1, 2, 3])))
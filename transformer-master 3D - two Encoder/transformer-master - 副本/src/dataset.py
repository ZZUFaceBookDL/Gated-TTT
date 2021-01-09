from torch.utils.data import Dataset
import torch

from scipy.io import loadmat


class OzeDataset(Dataset):
    def __init__(self, path, dataset):
        super(OzeDataset, self).__init__()
        self.dataset = dataset
        self.train_len,\
        self.test_len,\
        self.input_len,\
        self.channel_len,\
        self.output_len,\
        self.train_dataset,\
        self.train_label,\
        self.test_dataset,\
        self.test_label = self.pre_option(path)

    def __getitem__(self, index):
        if self.dataset == 'train':
          return self.train_dataset[index], self.train_label[index] - 1
          # if self.train_label[index] == 1:
          #   self.train_label[index] = self.train_label[index] - 1
          # elif self.train_label[index] == 13:
          #   self.train_label[index] = self.train_label[index] - 12
          # return self.train_dataset[index], self.train_label[index]
        elif self.dataset == 'test':
          return self.test_dataset[index], self.test_label[index] - 1
          # if self.test_label[index] == 1:
          #   self.test_label[index] = self.test_label[index] - 1
          # elif self.test_label[index] == 13:
          #   self.test_label[index] = self.test_label[index] - 12
          # return self.test_dataset[index], self.test_label[index]
          

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len

    def pre_option(self, path):
        m = loadmat(path)

        # m中是一个字典 有4个key 其中最后一个键值对存储的是数据
        x1, x2, x3, x4 = m
        data = m[x4]

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
        train_label = data00[index_trainlabels]
        train_data = data00[index_train]
        test_label = data00[index_testlabels]
        test_data = data00[index_test]

        train_label = train_label.squeeze()
        train_data = train_data.squeeze()
        test_label = test_label.squeeze()
        test_data = test_data.squeeze()

        train_len = train_data.shape[0]
        test_len = test_data.shape[0]
        output_len = len(tuple(set(train_label)))

        # 时间步最大值
        max_lenth = 0  # 93
        for item in train_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        for item in test_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        # train_data, test_data为numpy.object 类型，不能直接对里面的numpy.ndarray进行处理
        train_dataset = []
        test_dataset = []
        for x1 in train_data:
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            train_dataset.append(x1)

        for x2 in test_data:
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            test_dataset.append(x2)

        # 最后维度 [数据条数,时间步数最大值,时间序列维度]
        train_dataset = torch.stack(train_dataset, dim=0).permute(0, 2, 1)
        test_dataset = torch.stack(test_dataset, dim=0).permute(0, 2, 1)
        train_label = torch.Tensor(train_label)
        test_label = torch.Tensor(test_label)
        channel = test_dataset[0].shape[-1]
        input = test_dataset[0].shape[-2]

        return train_len, test_len, input, channel, output_len, train_dataset, train_label, test_dataset, test_label
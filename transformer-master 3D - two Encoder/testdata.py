class OzeDataset(Dataset):
    def __init__(self, filepath):
        super(OzeDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)
        # 此时取出来的数据是一个(N,M)的元组，N是行，M是列，xy.shape[0]可以把N取出来，也就是一共有多少行，有多少条数据
        self.len = xy.shape[0]
        # self.x_data = torch.from_numpy(xy[:, 1:])
        self.x_data = xy[:, 1:]

        self.y_data = xy[:, 0]
        # self.y_data = torch.from_numpy(xy[:, 0])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]-1

    def __len__(self):
        return self.len


train_path = 'F:\\PyChromProjects\\data\\UCRArchive_2018\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv'  # 数据集路径
test_path = 'F:\\PyChromProjects\\data\\UCRArchive_2018\\UCRArchive_2018\\ACSF1\\ACSF1_TEST.tsv'  # 数据集路径


dataset_train = OzeDataset(train_path)

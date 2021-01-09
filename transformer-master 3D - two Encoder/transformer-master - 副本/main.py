from torch.autograd import Variable

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

import numpy as np
import torch

from tst.transformer import Transformer

from src.sliding_window import sliding_window
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU, ConvGru, FFN
from src.metrics import MSE

import _pickle as cp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

NB_SENSOR_CHANNELS = 113
NUM_CLASSES = 18
SLIDING_WINDOW_LENGTH = 24
FINAL_SEQUENCE_LENGTH = 8
SLIDING_WINDOW_STEP = 12

NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128

#load sensor data

def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

#Segmentation and Reshaping


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


#创建子类
class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels=NUM_FILTERS, kernel_size = (5,1))
#         self.conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5,1))
#         self.conv3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5,1))
#         self.conv4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (5,1))
#
#         #self.fc1 = nn.Linear(57856, 128)
#         #self.fc2 = nn.Linear(128, 128)
#         self.lstm1 = nn.LSTM(input_size= (64*113), hidden_size=NUM_UNITS_LSTM, num_layers=1)
#         self.lstm2 = nn.LSTM(input_size=NUM_UNITS_LSTM, hidden_size=NUM_UNITS_LSTM, num_layers=1)
#         self.out = nn.Linear(128*8, NUM_CLASSES)
#
# #        self.fc3 = nn.Linear(84, NUM_CLASSES)
#
#     def forward(self, x):
#         x = (F.relu(self.conv1(x)))
#         x = (F.relu(self.conv2(x)))
#         x = (F.relu(self.conv3(x)))
#         x = (F.relu(self.conv4(x)))
#         x = x.permute(0, 2, 1, 3)
#         x = x.contiguous()
#         x = x.view(-1, 8, 64*113)
#
#         x = F.dropout(x, p=0.5)
#         #x = F.relu(self.fc1(x))
#
#         x, (h_n, c_n) = self.lstm1(x)
#         x = F.dropout(x, p=0.5)
#
#         #x = F.relu(self.fc2(x))
#         x, (h_n, c_n) = self.lstm2(x)
#         x = x.view(-1, 1 * 8 * 128)
#         x = F.dropout(x, p=0.5)
#         x = F.softmax(self.out(x), dim=1)
# #        x = F.relu(self.fc2(x))
# #        x = self.fc3(x)
#         return x

def my_loss(outputs, targets):
    output2 = outputs - torch.max(outputs, 1, True)[0]
    P = torch.exp(output2) / torch.sum(torch.exp(output2), 1,True) + 1e-10
    loss = -torch.mean(targets * torch.log(P))
    return loss


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def weights_init(m):
    
    # classname=m.__class__.__name__
    # if classname.find('conv') != -1:
    #     torch.nn.init.orthogonal(m.weight.data)
    #     torch.nn.init.orthogonal(m.bias.data)
    # if classname.find('lstm') != -1:
    #     torch.nn.init.orthogonal(m.weight.data)
    #     torch.nn.init.orthogonal(m.bias.data)
    # if classname.find('out') != -1:
    #     torch.nn.init.orthogonal(m.weight.data)
    #     torch.nn.init.orthogonal(m.bias.data)
    # if classname.find('fc') != -1:
    torch.nn.init.orthogonal(m.weight.data)
    torch.nn.init.orthogonal(m.bias.data)

draw_key = 1  # 大于等于draw_key才保存结果图
test_interval = 1  # 调用test()函数的epoch间隔
result_figure_path = 'result_view'  # 保存结果图像的路径
# lst_url = path.split('/')
# data_set_name = lst_url.pop(-2)
data_set_name = 'S1-ADL1'
optimizer_p = 'Adam'  # 优化器
d_model = 1024  # Lattent dim
q = 6  # Query size
v = 6  # Value size
h = 8  # Number of heads
N = 8  # Number of encoder and decoder to stack
dropout = 0.2  # Dropout rate
pe = True  # Positional encoding
mask = True
EPOCHS = 800
BATCH_SIZE = 70
LR = 1e-4

loss_list = []
correct_list = []
correct_list_ontrain = []

def test(net, dataloader_test, flag='test_set'):
    correct = 0.0
    total = 0
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.cuda(), y_test.cuda()

            test_outputs = net(enc_inputs)

            _, predicted = torch.max(test_outputs.data, dim=1)
            total += dec_inputs.size(0)
            correct += (predicted.float() == dec_inputs.squeeze(1).float()).sum().item()
        if flag == 'test_set':
            correct_list.append((100 * correct / total))
        elif flag == 'train_set':
            correct_list_ontrain.append((100 * correct / total))
        # tune.track.log(mean_accuracy=correct / total)
        print(f'Accuracy on {flag}: %f %%' % (100 * correct / total))

    # return correct / total


# 结果可视化 包括绘图和结果打印
def result_visualization():
    # my_font = fp(fname=r"C:\windows\Fonts\msyh.ttc")  # 2、设置字体路径

    # 设置风格
    # plt.style.use('ggplot')
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_list, color='red', label='on Test Dataset')
    ax2.plot(correct_list_ontrain, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'min_loss: {min(loss_list)}' '    '
                              # f'min_loss to epoch :{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}' '    '
                              f'last_loss:{loss_list[-1]}' '\n'
                              f'max_correct: {max(correct_list)}%' '    '
                              f'max_correct to training_epoch_number:{(correct_list.index(max(correct_list)) + 1) * test_interval}' '    '
                              f'last_correct: {correct_list[-1]}%' '\n'
                              f'd_model={d_model} dataset = {data_set_name}  q={q}   v={v}   h={h}   N={N} drop_out={dropout}' '\n'
             )

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCHS >= draw_key:
        plt.savefig(
            f'{result_figure_path}/{data_set_name}{max(correct_list)}% {optimizer_p} epoch={EPOCHS} batch={BATCH_SIZE} lr={LR} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()
    print('正确率列表', correct_list)

    print(f'最小loss：{min(loss_list)}\r\n'
          # f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：{max(correct_list)}\r\n'
          f'最大correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}\r\n'
          f'最大train correct：{max(correct_list_ontrain)}\r\n'
          f'最大train correct对应的已训练epoch数:{(correct_list_ontrain.index(max(correct_list_ontrain)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_list[-1]}')

    # print(f'共耗时{round(time_cost, 2)}分钟')

if __name__ == "__main__":

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')
    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    # Data is reshaped
    # X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv2D
    # X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv2D

    X_train = X_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv2D
    X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  # for input to Conv2D

    print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    d_input = SLIDING_WINDOW_LENGTH
    d_output = NUM_CLASSES
    d_channel = NB_SENSOR_CHANNELS

    # d_model = 512  # Lattent dim
    # q = 6  # Query size
    # v = 6  # Value size
    # h = 8  # Number of heads
    # N = 8  # Number of encoder and decoder to stack
    # dropout = 0.2  # Dropout rate
    # pe = True  # Positional encoding
    # mask = True

    net = Transformer(d_input, d_channel, d_model, d_output, q, v, h, N,
                       dropout=dropout, pe=pe).cuda()

    # optimizer, loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_F = torch.nn.CrossEntropyLoss()

    # create My Dateset

    train_set = subDataset(X_train, y_train)
    test_set = subDataset(X_test, y_test)

    print(train_set.__len__())
    print(test_set.__len__())

    trainloader = DataLoader.DataLoader(train_set, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=2)

    testloader = DataLoader.DataLoader(test_set, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=2)


    for idx_epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.long()

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            labels = labels.squeeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_F(outputs, labels)
            print('Epoch:', '%04d' % (idx_epoch + 1), 'loss =', '{:.6f}'.format(loss))
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
        if ((idx_epoch + 1) % test_interval) == 0:
            test(net, testloader)
            test(net, trainloader, 'train_set')
            print(f'max test accuracy: %f %% ,epoch = %d, d_model = %f ,lr = %f ,d_channel = %f, d_output = %f' % (max(correct_list), (correct_list.index(max(correct_list)) + 1) * test_interval, d_model, LR, d_channel,d_output))
            print(f'max train accuracy: %f %% ,epoch = %d, d_model = %f ,lr = %f ,d_channel = %f, d_output = %f' % (max(correct_list_ontrain), (correct_list_ontrain.index(max(correct_list_ontrain)) + 1) * test_interval,d_model, LR, d_channel, d_output))

            # # print statistics
            # running_loss += loss.item()
            # if i % 100 == 99:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0

        # corret, total = 0, 0
        # for datas, labels in testloader:
        #     datas = datas.cuda()
        #     labels = labels.cuda()
        #     outputs = net(datas)
        #     _, predicted = torch.max(outputs.data, 1)
        #
        #     labels = labels.long()
        #     total += labels.size(0)
        #     corret += (predicted == labels.squeeze(1)).sum()
        #
        #     predicted = predicted.cpu().numpy()
        #     labels = labels.cpu().squeeze(1).numpy()
        #     # F_score
        #     import sklearn.metrics as metrics
        #
        #     print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(labels, predicted, average='weighted')))


    print('Finished Training')
    # 调用结果可视化
    result_visualization()


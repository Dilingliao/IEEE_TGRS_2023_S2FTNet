from re import A
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import S2FTNet
from torchsummary import summary
import math
from sklearn import metrics, preprocessing
from functools import reduce 
import torch.utils.data as Data


import sys
from generate_pic import sampling, generate_png

def load_dataset(Dataset):
    if Dataset == 'IP':
        mat_data = sio.loadmat('F./datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('./datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.90
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('./datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('./datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):


    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)

    return small_cubic_data

def select_small_cubic_pixal(data_size, data_indices, whole_data, padded_data, dimension, patch_length=0):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    print(x_train.shape)
    print(band_patch)
    print(patch)
    nn = band_patch // 2
    print(nn)
    pp = (patch*patch) // 2
    print(pp)
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    print(x_train_reshape.shape)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    print(x_train_band.shape)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  whole_data_pca, whole_data, PATCH_LENGTH, padded_data_pca, padded_data, INPUT_DIMENSION_pca, INPUT_DIMENSION, 
                  batch_size, gt, PATCH,band, batch, band_patch=3):
    print(gt.shape)
    gt_all = gt[total_indices] - 1      # (10249,)     len: total_indices
    print(gt_all.shape)      
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data_pca, PATCH_LENGTH, padded_data_pca, INPUT_DIMENSION_pca)
    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data_pca, PATCH_LENGTH, padded_data_pca, INPUT_DIMENSION_pca)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data_pca, PATCH_LENGTH, padded_data_pca, INPUT_DIMENSION_pca)
    
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_pca)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_pca)
    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION_pca)


    all_data_pixal = select_small_cubic_pixal(TOTAL_SIZE, total_indices, whole_data, padded_data, INPUT_DIMENSION,  PATCH)
    train_data_pixal = select_small_cubic_pixal(TRAIN_SIZE, train_indices, whole_data, padded_data, INPUT_DIMENSION,  PATCH)
    test_data_pixal = select_small_cubic_pixal(TEST_SIZE, test_indices, whole_data, padded_data, INPUT_DIMENSION,  PATCH)

    train_data_pixal = gain_neighborhood_band(train_data_pixal, band, band_patch, batch )
    test_data_pixal = gain_neighborhood_band(test_data_pixal, band, band_patch, batch)
    all_data_pixal = gain_neighborhood_band(all_data_pixal, band,  band_patch, batch)

    x_train_pixal = train_data_pixal.reshape(train_data_pixal.shape[0], train_data_pixal.shape[1], INPUT_DIMENSION)
    x_test_all_pixal = test_data_pixal.reshape(test_data_pixal.shape[0], test_data_pixal.shape[1], INPUT_DIMENSION)
    all_data_pixal.reshape(all_data_pixal.shape[0], all_data_pixal.shape[1], INPUT_DIMENSION)

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_train = torch.from_numpy(x_train_pixal).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x2_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(x_test_all).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_test = torch.from_numpy(x_test_all_pixal).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, x2_tensor_test, y1_tensor_test)
    
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_pixal = torch.from_numpy(all_data_pixal).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_pixal, all_tensor_data_label)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  
        num_workers=0,  
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  
        num_workers=0,  
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  
        num_workers=0, 
    )

    return train_iter, test_iter, all_iter #, y_test

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def padWithZeros(X, margin):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels = True):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)    # (157, 157, 100)
    print("填充后：", zeroPaddedX.shape)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))    # (21025, 13, 13, 100)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))            # (21025, )  
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    print(patchesLabels.shape)
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 32


def create_data_loader():

    global Dataset
    dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
    Dataset = dataset.upper()
    X, y, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
    image_x, image_y, BAND = X.shape
    patch_size = 6
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)   # (145,145,200)
    print('Label shape: ', y.shape)   # (145,145)



    data = X.reshape(np.prod(X.shape[:2]), np.prod(X.shape[2:]))    # (21025, 200)
    gt = y.reshape(np.prod(y.shape[:2]),)     # (21025,)
    # print(gt.shape)

    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    PATCH_LENGTH = 6
    PATCH = 0
    batch_size = 64
    img=1

    img_rows = 2*PATCH_LENGTH+1
    img_cols = 2*PATCH_LENGTH+1
    img_channels = X.shape[2]
    INPUT_DIMENSION = X.shape[2]
    ALL_SIZE = X.shape[0] * X.shape[1]
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

    data = preprocessing.scale(data)
    data_ = data.reshape(X.shape[0], X.shape[1], X.shape[2])
    whole_data = data_
    padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                            'constant', constant_values=0)

    # pca后
    X_pca = applyPCA(X, numComponents=pca_components)    # (145,145,30) 
    data_pca = X_pca.reshape(np.prod(X_pca.shape[:2]), np.prod(X_pca.shape[2:]))    # (21025, 200)
    INPUT_DIMENSION_pca = X_pca.shape[2]

    data_pca = preprocessing.scale(data_pca)
    data__pca = data_pca.reshape(X_pca.shape[0], X_pca.shape[1], X_pca.shape[2])
    whole_data_pca = data__pca
    padded_data_pca = np.lib.pad(whole_data_pca, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                            'constant', constant_values=0)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)


    train_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  whole_data_pca, whole_data, PATCH_LENGTH, padded_data_pca, padded_data, INPUT_DIMENSION_pca, INPUT_DIMENSION, 
                  batch_size, gt, PATCH, BAND, img, band_patch=3)

    return train_iter, test_iter, all_iter, y, total_indices,  TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, img_rows*img_cols,img_channels,BAND


def train(train_loader, xy, img_channels, BAND, epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("training on ", device)
    torch.cuda.empty_cache()

    net= S2FTNet.S2FTNet(xy, BAND, img_channels).to(device)
    print(net)
    print('cliqueNet parameters:', sum(param.numel() for param in net.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    total_loss = 0

    for epoch in range(epochs):
        net.train()
        train_acc_sum, n = 0.0, 0
        for i, (data1, data2, target) in enumerate(train_loader):
            
            data1, target = data1.to(device), target.to(device)
            data2, target = data2.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data1,data2)
            # 计算损失函数
            # print(outputs.shape)
            # print(target.shape)
            loss = criterion(outputs, target.long())
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))

    return net, device




def test(device, net, test_loader, xta, xte, xall):
    count = 0
    y_pred_test = 0
    y_test = 0
    a = np.zeros([1000,])
    k=0

    for inputs1,inputs2, labels in test_loader:  # inputs.shape (64, 1, 100, 13, 13)   9225/64=144
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        net.eval()
        # print(inputs.shape)
        outputs = net(inputs1,inputs2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        # A = A.detach().cpu().numpy()

        # print( A)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
            # a = np.concatenate(A)


    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100




if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, gt_hsi, total_indices, xta, xte, xall, xy,img_channels,BAND = create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, xy, img_channels, BAND, epochs=200)

    torch.save(net.state_dict(),'./SSFTTnet_params_'+Dataset+'.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader, xta, xte, xall)
    # print('y_pred_test',y_pred_test)



    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = "./classification_report_IP.txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    generate_png(all_data_loader, net, gt_hsi, 'IP', device, total_indices)
    








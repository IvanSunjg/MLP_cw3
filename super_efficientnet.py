import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from generate_filter import Kernel
import os
import json
import time

LABEL = "Super_EffecientNet"

base_model = [
    # expand ratio, channels, repeats, stride, kernel size
    [1, 16, 1, 1, 3],
    [6, 24, 2 ,2 ,3], 
    [6, 40, 2, 2, 5], 
    [6, 80, 3, 2, 3],
    [6,112, 3, 1 ,5], 
    [6,192, 4, 2, 5],
    [6,320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0,224,0.2),  # alpha, beta, ganma, depth=alpha*phi
    "b1": (0.5,240,0.2),
    "b2": (1,260,0.3),
    "b3": (2,300,0.3),
    "b4": (3,380,0.4),
    "b5": (4,456,0.4),
    "b6": (5,528,0.5),
    "b7": (6,600,0.5),
}

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
        self.silu = nn.SiLU()
    
    def forward(self, x):
        x = self.silu(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self,x):
        #print(x.shape)
        #print(self.in_channels)
        x = self.cnn(x)
        x = self.bn(x)
        x = self.silu(x)

        return x

# Pooling part
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # (C,H,W) -> (C,1,1)
            nn.Conv2d(in_channels,reduced_dim,1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim,in_channels,1),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        return x*self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction = 4, survival_prob = 0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob # for stochastic depth
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size = 3, stride = 1, padding = 1,)

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training: #like a drop out, so not testing
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device) < self.survival_prob
        return torch.div(x, self.survival_prob)*binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class Super_EfficientNet(nn.Module):
    def __init__(self, version, filters, num_classes):
        super(Super_EfficientNet, self).__init__()
        self.filters = filters
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha = 1.2, beta = 1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [Activation(), CNNBlock(self.filters.shape[0]*3, channels, 3, stride = 2, padding = 1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(in_channels, out_channels, expand_ratio = expand_ratio, 
                    stride = stride if layer == 0 else 1, kernel_size = kernel_size, 
                    padding = kernel_size//2), #if k=1:pad=0, k=3:pad=1, k=5:pad=2
                )
                in_channels = out_channels

        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*features)

    def Red_Green(self,filter,image):
        batch_size = image.shape[0]
        r_filter = filter.view((filter.shape[0],1,filter.shape[1],filter.shape[1]))
        g_filter = -r_filter
        #print(r_filter.shape)
        #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
        new_r = F.conv2d(image[:,0].view(-1,1,image.shape[2],image.shape[3]), r_filter)
        new_g = F.conv2d(image[:,1].view(-1,1,image.shape[2],image.shape[3]), g_filter)
        return (new_r+new_g)

    def Blue_Yellow(self,filter,image):
        batch_size = image.shape[0]
        b_filter = filter.view((filter.shape[0],1,filter.shape[1],filter.shape[1]))
        y_filter = -b_filter
        #print(r_filter.shape)
        #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
        yellow = (image[:,0] + image[:,1]) / 2
        new_b = F.conv2d(image[:,2].view(-1,1,image.shape[2],image.shape[3]), b_filter)
        new_y = F.conv2d(yellow.view(-1,1,image.shape[2],image.shape[3]), y_filter)
        return (new_b+new_y)

    def Gray(self,filter,image):
        batch_size = image.shape[0]
        g_filter = filter.view((filter.shape[0],1,filter.shape[1],filter.shape[1]))
        #print(r_filter.shape)
        #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
        grey = (image[:,0] + image[:,1] + image[:,2]) / 3
        new_grey = F.conv2d(grey.view(-1,1,image.shape[2],image.shape[3]), g_filter)
        return new_grey

    def super_conv2d(self,filter,image):
        new_rg = self.Red_Green(filter,image)
        new_by = self.Blue_Yellow(filter,image)
        new_gray = self.Gray(filter,image)
        self.first_layer_img = torch.hstack((new_rg, new_by, new_gray))
        #print(self.first_layer_img.shape)
        return self.first_layer_img

    def forward(self,x):
        x = self.super_conv2d(self.filters,x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x

def plot_train(epoch,train_loss_list,test_loss_list,train_acc_list,test_acc_list):
    # visualizing training process
    plt.figure()
    plt.title("Train and Test Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1,epoch+1),train_loss_list,color="b",linestyle="-",label="train_loss")
    plt.plot(range(1,epoch+1),test_loss_list,color="r",linestyle="--",label="test_loss")
    plt.legend()
    plt.savefig("model_plots/"+LABEL+"_Train and Test Loss.png")

    plt.figure()
    plt.title("Train and Test Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(range(1,epoch+1),train_acc_list,color="b",linestyle="-",label="train_accuracy")
    plt.plot(range(1,epoch+1),test_acc_list,color="r",linestyle="--",label="test_accuracy")
    plt.legend()
    plt.savefig("model_plots/"+LABEL+"_Train and Test Accuracy.png")

def test_model(net,test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = 0.0
    test_acc = 0.0
    net.eval()  
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            y_pred_prob = net(test_images.to(device))
            y_pred = torch.max(y_pred_prob, dim=1)[1]
            test_acc += torch.sum(y_pred==test_labels.to(device)).item()/len(test_labels)
    acc = test_acc/len(test_loader)
    return acc

def main(epoch=100,lr=0.0002):
    #device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    version = "b0"
    phi, res, drop_rate = phi_values[version]

    # preparing data
    data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(res),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((res, res)),   # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((res, res)),   # cannot 224, must (224, 224)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                    }

    batch_size = 8
    train_dataset = datasets.ImageFolder(root="data/train", transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    validate_dataset = datasets.ImageFolder(root="data/val", transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = datasets.ImageFolder(root="data/test", transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    cd_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cd_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    kernels = Kernel()().to(device)
    net = Super_EfficientNet(filters=kernels,version=version,num_classes=len(cla_dict))
    net.to(device)
    print(net)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    save_path = './Super_EfficientNet.pth'
    best_acc = 0.0

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for i in range(epoch):
        print("\n------------------------------------------------------")
        print("epoch {}/{}".format(i+1,epoch))
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        net.train()
        t1 = time.perf_counter()
        for data in train_loader:
            images, labels = data
            optimizer.zero_grad()
            y_pred_prob = net(images.to(device))
            loss = loss_function(y_pred_prob, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred = torch.max(y_pred_prob,1)[1]                        #[1] is indicies of the resulting tensor
            train_acc += torch.sum(y_pred==labels.to(device)).item()/len(labels)
        batch_train_loss = train_loss/len(train_loader)
        batch_train_acc = train_acc/len(train_loader)
        print("Time taken for Training: ",time.perf_counter()-t1)

        # validate
        net.eval()  
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                y_pred_prob = net(val_images.to(device))
                loss = loss_function(y_pred_prob,val_labels.to(device))
                val_loss += loss.item()
                y_pred = torch.max(y_pred_prob, dim=1)[1]
                val_acc += torch.sum(y_pred==val_labels.to(device)).item()/len(val_labels)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), save_path)
        batch_test_loss = val_loss/len(validate_loader)
        batch_test_acc = val_acc/len(validate_loader)
        print("Train_Loss: {:.4f} Train_Acc: {:.4f}".format(batch_train_loss, batch_train_acc))
        print("Test_Loss: {:.4f} Test_Acc: {:.4f}".format(batch_test_loss,batch_test_acc))
        
        train_loss_list.append(batch_train_loss)
        train_acc_list.append(batch_train_acc)
        test_loss_list.append(batch_test_loss)
        test_acc_list.append(batch_test_acc)
    
    print('Finished Training')
    return net, train_loss_list, train_acc_list, test_loss_list, test_acc_list, test_loader


if __name__ == '__main__':
    import sys
    try:
        epoch = int(sys.argv[1])
    except:
        epoch = 100
    try:
        lr = float(sys.argv[2])
    except:
        lr = 0.0002
    # strat training
    net, train_loss_list, train_acc_list, test_loss_list, test_acc_list, test_loader = main(epoch=epoch,lr=lr)
    # plot the training process
    plot_train(epoch, train_loss_list, test_loss_list, train_acc_list, test_acc_list)
    # test results
    print('Test accuracy: ',test_model(net,test_loader))
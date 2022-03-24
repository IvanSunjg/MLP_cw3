import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from generate_filter import Kernel
import os
import json
import time

LABEL = "Super_AlexNet"

class Super_AlexNet(nn.Module):
    def __init__(self, filters, num_classes=1000, mid_units=100, init_weights=False):   
        super(Super_AlexNet, self).__init__()
        self.filters = filters
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.Conv2d(filters.shape[0]*3, mid_units, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True), #inplace
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(mid_units, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.super_conv2d(self.filters,x)
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def plot_train(epoch,train_loss_list,test_loss_list,train_acc_list,test_acc_list,mid_units):
    # visualizing training process
    plt.figure()
    plt.title("Train and Test Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1,epoch+1),train_loss_list,color="b",linestyle="-",label="train_loss")
    plt.plot(range(1,epoch+1),test_loss_list,color="r",linestyle="--",label="test_loss")
    plt.legend()
    plt.savefig("model_plots/"+LABEL+"_"+str(mid_units)+"_Train and Test Loss.png")

    plt.figure()
    plt.title("Train and Test Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(range(1,epoch+1),train_acc_list,color="b",linestyle="-",label="train_accuracy")
    plt.plot(range(1,epoch+1),test_acc_list,color="r",linestyle="--",label="test_accuracy")
    plt.legend()
    plt.savefig("model_plots/"+LABEL+"_"+str(mid_units)+"_Train and Test Accuracy.png")

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

def super_alexnet_main(mid_units,epoch=300,lr=0.0002):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # preparing data
    data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),   # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((224, 224)),   # cannot 224, must (224, 224)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                    }

    batch_size = 8
    train_dataset = datasets.ImageFolder(root="../data/train", transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    validate_dataset = datasets.ImageFolder(root="../data/val", transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = datasets.ImageFolder(root="../data/test", transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    cd_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cd_list.items())

    # strat training
    kernels = Kernel()().to(device) 
    net = Super_AlexNet(kernels, num_classes=len(cla_dict), mid_units=mid_units, init_weights=False)
    net.to(device)
    #print(net)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    save_path = '.params/Super_AlexNet_'+str(mid_units)+'.pth'
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
    mid_units = int(sys.argv[1])
    try:
        epoch = int(sys.argv[2])
    except:
        epoch = 100
    try:
        lr = float(sys.argv[3])
    except:
        lr = 0.0002
    # strat training
    net, train_loss_list, train_acc_list, test_loss_list, test_acc_list, test_loader = super_alexnet_main(mid_units,epoch=epoch,lr=lr)
    # plot the training process
    plot_train(epoch, train_loss_list, test_loss_list, train_acc_list, test_acc_list, mid_units)
    # test results
    print('Test accuracy for mid_units = '+str(mid_units)+': ',test_model(net,test_loader))

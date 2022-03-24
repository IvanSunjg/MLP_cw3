import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

LABEL = "Pretrained_AlexNet"

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

def pretrained_alexnet_main():
    #device : GPU or CPU
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
    
    # strat training 
    net = models.alexnet(pretrained=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.0002
    optimizer = optim.Adam(net.parameters(), lr=lr)
    save_path = './params/pre_trained_AlexNet.pth'
    best_acc = 0.0
    epoch = 300

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

    # plot the training process
    plot_train(epoch, train_loss_list, test_loss_list, train_acc_list, test_acc_list)
    # test results
    print('Test accuracy: ',test_model(net,test_loader))

if __name__ == '__main__':
    pretrained_alexnet_main()
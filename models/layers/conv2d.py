import torch
import torch.nn as nn
import torch.nn.functional as F

def Red_Green(filter,image):
    batch_size = image.shape[0]
    r_filter = filter.view((400,1,filter.shape[1],filter.shape[1]))
    g_filter = -r_filter
    #print(r_filter.shape)
    #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
    new_r = F.conv2d(image[:,0].view(-1,1,image.shape[2],image.shape[3]), r_filter)
    new_g = F.conv2d(image[:,1].view(-1,1,image.shape[2],image.shape[3]), g_filter)
    return (new_r+new_g)

def Blue_Yellow(filter,image):
    batch_size = image.shape[0]
    b_filter = filter.view((400,1,filter.shape[1],filter.shape[1]))
    y_filter = -b_filter
    #print(r_filter.shape)
    #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
    yellow = (image[:,0] + image[:,1]) / 2
    new_b = F.conv2d(image[:,2].view(-1,1,image.shape[2],image.shape[3]), b_filter)
    new_y = F.conv2d(yellow.view(-1,1,image.shape[2],image.shape[3]), y_filter)
    return (new_b+new_y)

def Gray(filter,image):
    batch_size = image.shape[0]
    g_filter = filter.view((400,1,filter.shape[1],filter.shape[1]))
    #print(r_filter.shape)
    #print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
    grey = (image[:,0] + image[:,1] + image[:,2]) / 3
    new_grey = F.conv2d(grey.view(-1,1,image.shape[2],image.shape[3]), g_filter)
    return new_grey

def super_conv2d(filter,image):
    new_rg = Red_Green(filter,image)
    new_by = Blue_Yellow(filter,image)
    new_gray = Gray(filter,image)
    return torch.hstack((new_rg, new_by, new_gray))

if __name__ == '__main__':
    #device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image = torch.randn((10,3,128,128)).to(device)
    filter1 = torch.rand((400,3,3)).to(device)

    """ new_img = Red_Green(filter1,image)
    print(new_img.shape)
    new_img = Blue_Yellow(filter1,image)
    print(new_img.shape)
    new_img = Gray(filter1,image)
    print(new_img.shape) """

    new_image = super_conv2d(filter1,image)
    print(new_image.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

def Red_Green(filter,image):
    batch_size = image.shape[0]
    r_filter = filter + torch.zeros((1,1,filter.shape[0],filter.shape[1])).to(device)
    g_filter = -r_filter
    print(r_filter.shape)
    print(image[:,0].view(-1,1,image.shape[2],image.shape[3]).shape)
    new_r = F.conv2d(image[:,0].view(-1,1,image.shape[2],image.shape[3]), r_filter)
    new_g = F.conv2d(image[:,1].view(-1,1,image.shape[2],image.shape[3]), g_filter)
    return (new_r+new_g)[:,0]

if __name__ == '__main__':
    #device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image = torch.rand((10,3,128,128)).to(device)
    filter1 = torch.rand((3,3)).to(device)

    new_img = Red_Green(filter1,image)
    print(new_img.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F


class Super_AlexNet(nn.Module):
    def __init__(self, filters, num_classes=1000, init_weights=False):   
        super(Super_AlexNet, self).__init__()
        self.filters = filters
        self.features = nn.Sequential(  
            nn.Conv2d(filters.shape[0]*3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True), #inplace
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
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
            nn.Linear(128 * 4 * 4, 2048),
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    images = torch.randn((8,3,224,224)).to(device)
    image_np = images[0].permute(1,2,0).cpu().detach().numpy()
    plt.imshow(image_np)
    plt.show()
    filters = torch.rand((404,45,45)).to(device)

    model = Super_AlexNet(filters)
    model.to(device)
    print(images.shape)
    y = model(images)
    print(y.shape)

    image1_np = model.first_layer_img[0,0].cpu().detach().numpy()
    plt.imshow(image1_np)
    plt.show()
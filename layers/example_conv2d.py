import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from typing import Optional, List, Tuple, Union
from torch import Tensor

class Red_Green(nn.Conv2d):
    
    def __init__(self, filter, **kwargs):
        super(Red_Green, self).__init__(kernel_size=filter.shape[0], **kwargs)
        print(self.weight.shape)
        self.weight = nn.Parameter(filter)
        self.weight.require_grad = False
        self.stride = 3
    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        print(self.stride)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        result = self._conv_forward(inputs, self.weight, self.bias)
        return result

if __name__ == '__main__':
    #device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image = torch.rand((3,128,128)).to(device)
    #image_np = image.cpu().detach().numpy().T
    #print(image_np.shape)
    #cv2.imshow("image",image_np)
    #cv2.waitKey(0)
    #cv2.destroyAllwindows()

    filter = torch.rand((3,3,3))
    conv = Red_Green(in_channels=3, out_channels=1, filter=filter)
    new_image = conv(image)
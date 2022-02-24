import numpy as np
import torch
from skimage.filters import gabor_kernel


class Gabor():

    def __init__(self) -> None:
        pass

    def __call__(self) -> torch.Tensor:
        
        """Gabor Filter"""

        t_1 = torch.from_numpy(np.real(gabor_kernel(frequency=0.08, theta=0, sigma_x=None, sigma_y=None, n_stds=3, offset=0)))
        t_2 = torch.from_numpy( - np.real(gabor_kernel(frequency=0.08, theta=0, sigma_x=None, sigma_y=None, n_stds=3, offset=0)))
        t = torch.stack((t_1,t_2))

        for theta in range(1,4):
            theta = theta / 4. * np.pi
            kernel = np.real(gabor_kernel(frequency=0.08, theta=theta, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
            kernel = np.array([np.pad(kernel,int((45-kernel.shape[0])/2),mode='constant')])
            t = torch.cat((t,torch.from_numpy(kernel)))
            t = torch.cat((t,torch.from_numpy(-kernel)))

        for theta in range(4):
            theta = theta / 4. * np.pi
            for frequency in (0.14,0.23,0.40):
                kernel = torch.from_numpy(np.real(gabor_kernel(frequency, theta=theta, sigma_x=None, sigma_y=None, n_stds=3, offset=0)))
                kernel = np.array([np.pad(kernel,int((45-kernel.shape[0])/2),mode='constant')])
                t = torch.cat((t,torch.from_numpy(kernel)))
                t = torch.cat((t,torch.from_numpy(-kernel)))


        return t

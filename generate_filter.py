import numpy as np
import torch
import cv2
from skimage.filters import gabor_kernel, gaussian


class Kernel():

    def __init__(self) -> None:
        pass

    def __call__(self) -> torch.Tensor:
        
        """Gabor Filter"""

        t_1 = np.real(gabor_kernel(frequency=0.08, theta=0, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
        t_1 = np.array([np.pad(t_1,int((49-t_1.shape[0])/2),mode='constant')])
        t_1 = torch.from_numpy(t_1)
        t = torch.cat((t_1,- t_1))

        for theta in range(1,4):
            theta = theta / 4. * np.pi
            kernel = np.real(gabor_kernel(frequency=0.08, theta=theta, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
            kernel = np.array([np.pad(kernel,int((49-kernel.shape[0])/2),mode='constant')])
            t = torch.cat((t,torch.from_numpy(kernel)))
            t = torch.cat((t,torch.from_numpy(-kernel)))

        for theta in range(4):
            theta = theta / 4. * np.pi
            for frequency in (0.14,0.23,0.40):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
                kernel = np.array([np.pad(kernel,int((49-kernel.shape[0])/2),mode='constant')])
                t = torch.cat((t,torch.from_numpy(kernel)))
                t = torch.cat((t,torch.from_numpy(-kernel)))

        """Gabor Half Bar Filter"""

        for theta in range(8):
            theta = theta / 4. * np.pi
            for frequency in (0.08,0.14,0.23,0.40):
                kernel = self.gabor_half(np.real(gabor_kernel(frequency, theta=theta, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=theta)
                kernel = np.array([np.pad(kernel,int((49-kernel.shape[0])/2),mode='constant')])
                t = torch.cat((t,torch.from_numpy(kernel)))
                t = torch.cat((t,torch.from_numpy(-kernel)))

        """Blob Filter"""

        for sigma in [1,2,4,8]:
            n = 5
            while True:
                m = np.zeros([n,n])
                m[int((n-1)/2)][int((n-1)/2)] = 1
                g1 = gaussian(m,sigma=sigma)
                g2 = gaussian(m,sigma=1.6*sigma)
                if (sum(sum(abs(g1))) >= 0.99 and sum(sum(abs(g2))) >= 0.99) or n >= 49:
                    kernel = torch.from_numpy(g1-g2)
                    kernel = np.array([np.pad(kernel,int((49-kernel.shape[0])/2),mode='constant')])
                    t = torch.cat((t,torch.from_numpy(kernel)))
                    t = torch.cat((t,torch.from_numpy(-kernel)))
                    break
                n = n + 2


        """Edge Filter"""

        for i in [0,45,90,135,180,225,270,315]:
            for sigma in [1,2,4,8]:
                kernel = self.edge(i,sigma)
                kernel = np.array([np.pad(kernel,int((49-kernel.shape[0])/2),mode='constant')])
                t = torch.cat((t,torch.from_numpy(kernel)))
                t = torch.cat((t,torch.from_numpy(-kernel)))

        return t.float()
    

    """Setting Gabor Half Bar kernel"""
    def gabor_half(self,kernel,theta):
        length = kernel.shape[0]
        if theta == 0:
            for k in range(length):
                kernel[int((length-1)/2)][k] = 0
            for i in range(int((length-1)/2)+1,length):
                for j in range(length):
                    kernel[i][j] = -kernel[i][j]
        elif theta == np.pi/4 :
            for i in range(length):
                for j in range(i+1):
                    if i == j:
                        kernel[i][j] = 0
                    else:
                        kernel[i][j] = - kernel[i][j]
        elif theta == np.pi/2 :
            for i in range(length):
                kernel[i][int((length-1)/2)] = 0
                for j in range(int((length-1)/2)):
                    kernel[i][j] = - kernel[i][j]
        elif theta == 3*np.pi/4 :
            for k in range(length):
                kernel[k][length-k-1] = 0
            for i in range(length):
                for j in range(length - i):
                    kernel[i][j] = - kernel[i][j]
        elif theta == np.pi :
            for k in range(length):
                kernel[int((length-1)/2)][k] = 0
            for i in range(int((length-1)/2)):
                for j in range(length):
                    kernel[i][j] = - kernel[i][j]
        elif theta == 5*np.pi/4:
            for i in range(length):
                for j in range(i,length):
                    if i == j:
                        kernel[i][j] = 0
                    else:
                        kernel[i][j] = - kernel[i][j]
        elif theta == 3*np.pi/2:
            for i in range(length):
                kernel[i][int((length-1)/2)] = 0
                for j in range(int((length-1)/2)+1,length):   
                    kernel[i][j] = - kernel[i][j]
        else:
            for k in range(length):
                kernel[k][length-k-1] = 0
            for i in range(length):
                for j in range(length-i,length):
                    kernel[i][j] = - kernel[i][j]
        return kernel

    """Setting Edge Kernel"""
    def edge(self,i,sigma):
        n = 5
        if i == 0:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+1)/2)][int((n+3)/2)] = 1
                    E[int((n+5)/2)][int((n+3)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2 
        elif i == 45:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+1)/2)][int((n+5)/2)] = 1
                    E[int((n+5)/2)][int((n+1)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 90:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+3)/2)][int((n+5)/2)] = 1
                    E[int((n+3)/2)][int((n+1)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 135:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+5)/2)][int((n+5)/2)] = 1
                    E[int((n+1)/2)][int((n+1)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 180:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+5)/2)][int((n+3)/2)] = 1
                    E[int((n+1)/2)][int((n+3)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 225:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+5)/2)][int((n+1)/2)] = 1
                    E[int((n+1)/2)][int((n+5)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 270:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+3)/2)][int((n+1)/2)] = 1
                    E[int((n+3)/2)][int((n+5)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        elif i == 315:
            while True:
                m1 = np.zeros([n,n])
                m1[int((n-1)/2)][int((n-1)/2)] = 1
                g = gaussian(m1,sigma=sigma)
                if sum(sum(abs(g))) >= 0.99:
                    g = g/sum(sum(abs(g)))
                    E = np.zeros([n+4,n+4])
                    E[int((n+1)/2)][int((n+1)/2)] = 1
                    E[int((n+5)/2)][int((n+5)/2)] = - 1
                    E1 = cv2.filter2D(E,-1,g)
                    break
                n = n + 2
        return E1

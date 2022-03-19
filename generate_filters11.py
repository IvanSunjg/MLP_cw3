import numpy as np
import torch
import cv2
from skimage.filters import gabor_kernel, gaussian

class Kernel():

    def __init__(self) -> None:
        pass

    def __call__(self) -> torch.Tensor:
        
        """Gabor Filter  X 4   """
        t_1 = np.real(gabor_kernel(frequency=0.40, theta=0, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
        t_1 = torch.from_numpy(t_1)

        t_2 = np.real(gabor_kernel(frequency=0.24, theta=1 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
        t_2 = torch.from_numpy(t_2)

        t = torch.stack((t_1, t_2))

        t_3 = np.real(gabor_kernel(frequency=0.40, theta=1 / 2. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
        t_3 = torch.from_numpy(np.array([t_3]))
        t = torch.cat((t,t_3))

        t_4 = np.real(gabor_kernel(frequency=0.24, theta=3 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0))
        t_4 = torch.from_numpy(np.array([t_4]))
        t = torch.cat((t,t_4))

        """Gabor Filter Half Bar  X 8   """
        t_5 = self.gabor_half(np.real(gabor_kernel(frequency=0.40, theta=0, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=0)
        t_5 = torch.from_numpy(np.array([t_5]))
        t = torch.cat((t,t_5))

        t_6 = self.gabor_half(np.real(gabor_kernel(frequency=0.24, theta=1 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=1 / 4. * np.pi)
        t_6 = torch.from_numpy(np.array([t_6]))
        t = torch.cat((t,t_6))

        t_7 = self.gabor_half(np.real(gabor_kernel(frequency=0.40, theta=1 / 2. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=1 / 2. * np.pi)
        t_7 = torch.from_numpy(np.array([t_7]))
        t = torch.cat((t,t_7))

        t_8 = self.gabor_half(np.real(gabor_kernel(frequency=0.24, theta=3 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=3 / 4. * np.pi)
        t_8 = torch.from_numpy(np.array([t_8]))
        t = torch.cat((t,t_8))

        t_9 = self.gabor_half(np.real(gabor_kernel(frequency=0.40, theta=np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=np.pi)
        t_9 = torch.from_numpy(np.array([t_9]))
        t = torch.cat((t,t_9))

        t_10 = self.gabor_half(np.real(gabor_kernel(frequency=0.24, theta=5 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=5 / 4. * np.pi)
        t_10 = torch.from_numpy(np.array([t_10]))
        t = torch.cat((t,t_10))

        t_11 = self.gabor_half(np.real(gabor_kernel(frequency=0.40, theta=3 / 2. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=3 / 2. * np.pi)
        t_11 = torch.from_numpy(np.array([t_11]))
        t = torch.cat((t,t_11))

        t_12 = self.gabor_half(np.real(gabor_kernel(frequency=0.24, theta=7 / 4. * np.pi, sigma_x=None, sigma_y=None, n_stds=3, offset=0)),theta=7 / 4. * np.pi)
        t_12 = torch.from_numpy(np.array([t_12]))
        t = torch.cat((t,t_12))

        """Blob Filter  X 4  """

        for sigma in [1,2,4,8]:
            n = 11
            m = np.zeros([n,n])
            m[int((n-1)/2)][int((n-1)/2)] = 1
            g1 = gaussian(m,sigma=sigma)
            g2 = gaussian(m,sigma=1.6*sigma)
            kernel = torch.from_numpy(np.array([g1-g2]))
            t = torch.cat((t,kernel))
        
        """Edge Filter  X 8  """

        for i in [0,45,90,135,180,225,270,315]:
            sigma = 2
            kernel = self.edge(i,sigma)
            t = torch.cat((t,torch.from_numpy(np.array([kernel]))))

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
        n = 7
        if i == 0:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+1)/2)][int((n+3)/2)] = 1
            E[int((n+5)/2)][int((n+3)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 45:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+1)/2)][int((n+5)/2)] = 1
            E[int((n+5)/2)][int((n+1)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 90:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+3)/2)][int((n+5)/2)] = 1
            E[int((n+3)/2)][int((n+1)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 135:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+5)/2)][int((n+5)/2)] = 1
            E[int((n+1)/2)][int((n+1)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 180:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+5)/2)][int((n+3)/2)] = 1
            E[int((n+1)/2)][int((n+3)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 225:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+5)/2)][int((n+1)/2)] = 1
            E[int((n+1)/2)][int((n+5)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 270:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+3)/2)][int((n+1)/2)] = 1
            E[int((n+3)/2)][int((n+5)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)

        elif i == 315:
            m1 = np.zeros([n,n])
            m1[int((n-1)/2)][int((n-1)/2)] = 1
            g = gaussian(m1,sigma=sigma)
            g = g/sum(sum(abs(g)))
            E = np.zeros([n+4,n+4])
            E[int((n+1)/2)][int((n+1)/2)] = 1
            E[int((n+5)/2)][int((n+5)/2)] = - 1
            E1 = cv2.filter2D(E,-1,g)
        
        return E1
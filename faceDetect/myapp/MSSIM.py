import numpy as np
from scipy.signal import convolve2d
import math

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim1(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    C3=C2/2  #求C3
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')  #图片1均值
    mu2 = filter2(im2, window, 'valid')  #图片2均值
    mu1_sq = mu1 * mu1  #图片1均值平方
    mu2_sq = mu2 * mu2  #图片1均值平方
    mu1_mu2 = mu1 * mu2  #图片1均值#图片1均值乘积
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq  #图片1方差
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq  #图片2方差
    sigma12 = filter2(im1*im2, window, 'valid') - mu1_mu2  #图片协方差
    #计算方差
    sigma1=np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)

    # ssim_map = ((2*mu1_mu2+C1) * (2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1*sigma2 + C2)) *(sigma12+C3)/ ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)*(sigma1*sigma2+C3))

    return np.mean(np.mean(ssim_map))

def compute_ssim2(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    C3=C2/2  #求C3
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')  #图片1均值
    mu2 = filter2(im2, window, 'valid')  #图片2均值
    mu1_sq = mu1 * mu1  #图片1均值平方
    mu2_sq = mu2 * mu2  #图片1均值平方
    mu1_mu2 = mu1 * mu2  #图片1均值#图片1均值乘积
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq  #图片1方差
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq  #图片2方差
    sigma12 = filter2(im1*im2, window, 'valid') - mu1_mu2  #图片协方差
    #计算方差
    sigma1=np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)

    ssim_map = ((2*mu1_mu2+C1) * (2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1*sigma2 + C2)) *(sigma12+C3)/ ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)*(sigma1*sigma2+C3))

    return np.mean(np.mean(ssim_map))
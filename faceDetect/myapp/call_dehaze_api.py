import cv2
import numpy as np
import math
from myapp import MSSIM
import traceback
import logging

logging.basicConfig(level=logging.INFO)

class Dehaze(object):
    def __init__(self,r):
        self.logger = logging.getLogger("Django")
        self.r = r

    def zmMinFilterGray(self,src, r=7):
        '''最小值滤波，r是滤波器半径'''
        try:
            if r <= 0:
                return src
            h, w = src.shape[:2]
            I = src
            res = np.minimum(I, I[[0] + [x for x in range(h - 1)], :])
            res = np.minimum(res, I[[x for x in range(1, h)] + [h - 1], :])
            I = res
            res = np.minimum(I, I[:, [0] + [x for x in range(w - 1)]])
            res = np.minimum(res, I[:, [x for x in range(1, w)] + [w - 1]])
            return self.zmMinFilterGray(res, r - 1)
        except Exception as e:
            traceback.print_exc()

    def guidedfilter(self,I, p, r, eps):
        '''引导滤波，直接参考网上的matlab代码'''
        try:
            height, width = I.shape
            m_I = cv2.boxFilter(I, -1, (r, r))
            m_p = cv2.boxFilter(p, -1, (r, r))
            m_Ip = cv2.boxFilter(I * p, -1, (r, r))
            cov_Ip = m_Ip - m_I * m_p

            m_II = cv2.boxFilter(I * I, -1, (r, r))
            var_I = m_II - m_I * m_I

            a = cov_Ip / (var_I + eps)
            b = m_p - a * m_I

            m_a = cv2.boxFilter(a, -1, (r, r))
            m_b = cv2.boxFilter(b, -1, (r, r))
            return m_a * I + m_b
        except Exception as e:
            traceback.print_exc()

    def getV1(self,m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        try:
            V1 = np.min(m, 2)  # 得到暗通道图像
            V1 = self.guidedfilter(V1, self.zmMinFilterGray(V1, self.r), r, eps)  # 使用引导滤波优化
            bins = 2000
            ht = np.histogram(V1, bins)  # 计算大气光照A
            d = np.cumsum(ht[0]) / float(V1.size)
            for lmax in range(bins - 1, 0, -1):
                if d[lmax] <= 0.999:
                    break
            A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

            V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

            return V1, A
        except Exception as e:
            traceback.print_exc()

    def deHaze(self,m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        try:
            print("--------开始去雾-----------")
            Y = np.zeros(m.shape)
            V1, A = self.getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
            for k in range(3):
                Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
            Y = np.clip(Y, 0, 1)
            if bGamma:
                Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
            return Y
        except Exception as e:
            traceback.print_exc()

    # 计算PSNR  不分通道
    def compute_psnr1(self,Im11, Im12):
        print("----------计算psnr---------")
        diff = Im11 - Im12
        rmse = math.sqrt(np.mean(diff ** 2.))  # 此结果为三维n*m矩阵？是一个float数
        # rmse=rmse[:, :, 0]+rmse[:, :, 1]+rmse[:, :, 2]
        return 20 * math.log10(255.0 / rmse)

    # 计算PSNR 分R,G,B三个通道
    def psnr2(self,Im1, Im2):
        # 导入你要测试的图像
        Im1 = np.array(Im1, 'f')  # 将图像1数据转换为float型
        Im1 = np.array(Im1, 'f')  # 将图像2数据转换为float型
        # 图像的行数
        height = Im1.shape[0]
        # 图像的列数
        width = Im1.shape[1]
        # 图像1,2各自分量相减，然后做平方；
        R = Im1[:, :, 0] - Im2[:, :, 0]
        G = Im1[:, :, 1] - Im2[:, :, 1]
        B = Im1[:, :, 2] - Im2[:, :, 2]
        # 做平方
        mser = R * R
        mseg = G * G
        mseb = B * B
        # 三个分量差的平方求和
        SUM = mser.sum() + mseg.sum() + mseb.sum()
        MSE = SUM / (height * width * 3)
        PSNR = 10 * math.log((255.0 * 255.0 / (MSE)), 10)
        return PSNR

    # 求灰度图像
    def rgb2gray(self,rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # dot--矩阵相乘，彩色图像与灰度图像关系：Y = 0.3R + 0.59G + 0.11B

    # 求灰度图像
    def rgb2gray(self,rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # dot--矩阵相乘，彩色图像与灰度图像关系：Y = 0.3R + 0.59G + 0.11B


    def compute_mssim(self,im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
        im1_gray = self.rgb2gray(im1)
        im2_gray=self.rgb2gray(im2)
        mssim = MSSIM.compute_ssim1(im1_gray,im2_gray)
        return mssim

    def test(self,x):
        try:
            return x*x
        except Exception as e:
            traceback.print_exc()


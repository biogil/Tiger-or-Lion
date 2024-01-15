import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def load_data(filename="datasets/", data_num=500):
    x_train = []
    t_train = []

    x_test = []
    t_test = []

    for i in range(data_num):
        im1 = Image.open(filename + "Lion/" + str(i + 1) + ".png")
        im2 = Image.open(filename + "Tiger/" + str(i + 1) + ".png")

        if im1.mode == "RGBA":
            im1 = im1.convert("RGB")
        if im2.mode == "RGBA":
            im2 = im2.convert("RGB")

        im1 = im1.resize((56, 56))
        im2 = im2.resize((56, 56))

        im1_a = np.array(im1)
        im2_a = np.array(im2)

        if i < data_num * 0.7:
            x_train.append(im1_a.transpose(2, 1, 0))
            t_train.append([1, 0])
            x_train.append(im2_a.transpose(2, 1, 0))
            t_train.append([0, 1])
        else:
            x_test.append(im1_a.transpose(2, 1, 0))
            t_test.append([1, 0])
            x_test.append(im2_a.transpose(2, 1, 0))
            t_test.append([0, 1])

    return (np.array(x_train), np.array(t_train)), (np.array(x_test), np.array(t_test))


def image_reversal(img, savefilepath, save_filename):
    """ 图像翻转"""
    lr = img.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
    ud = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
    lr.save(savefilepath + save_filename)
    ud.save(savefilepath + save_filename)


def image_rotation(img, savefilepath, save_filename):
    """图像旋转"""
    out1 = img.rotate(40)  # 旋转20度
    out2 = img.rotate(30)  # 旋转30度
    out1.save(savefilepath + save_filename)
    out2.save(savefilepath + save_filename)


def image_translation(img, savefilepath, save_filename):
    """图像平移"""
    out3 = ImageChops.offset(img, 20, 0)  # 只沿X轴平移
    out4 = ImageChops.offset(img, 0, 20)  # 只沿y轴平移
    out3.save(savefilepath + save_filename)
    out4.save(savefilepath + save_filename)


def image_brightness(img, savefilepath, save_filename):
    """亮度调整"""
    bri = ImageEnhance.Brightness(img)
    bri_img1 = bri.enhance(0.8)  # 小于1为减弱
    bri_img2 = bri.enhance(1.2)  # 大于1为增强
    bri_img1.save(savefilepath + save_filename)
    bri_img2.save(savefilepath + save_filename)


def image_chroma(img, savefilepath, save_filename):
    """色度调整"""
    col = ImageEnhance.Color(img)
    col_img1 = col.enhance(0.7)  # 色度减弱
    col_img2 = col.enhance(1.3)  # 色度增强
    col_img1.save(savefilepath + save_filename)
    col_img2.save(savefilepath + save_filename)


def image_contrast(img, savefilepath, save_filename):
    """对比度调整"""
    con = ImageEnhance.Contrast(img)
    con_img1 = con.enhance(0.7)  # 对比度减弱
    con_img2 = con.enhance(1.3)  # 对比度增强
    con_img1.save(savefilepath + save_filename)
    con_img2.save(savefilepath + save_filename)


def image_sharpness(img, savefilepath, save_filename):
    """锐度调整"""
    sha = ImageEnhance.Sharpness(img)
    sha_img1 = sha.enhance(0.5)  # 锐度减弱
    sha_img2 = sha.enhance(1.5)  # 锐度增强
    sha_img1.save(savefilepath + save_filename)
    sha_img2.save(savefilepath + save_filename)

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 损失函数
def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))


# 梯度下降法
class SGD:  # 随机梯度下降法
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_dict(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:  # 动量法
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update_dict(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class AdaGrad:  # 自适应法
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update_dict(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam:  # 自适应动量法
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update_dict(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

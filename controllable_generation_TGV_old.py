import functools
import time

import torch
from numpy.testing._private.utils import measure
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import identity
import scipy.io as sio
from PIL import Image
from utils import *
import time, itertools
from models import utils as mutils
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
from utils import fft2, ifft2, fft2_m, ifft2_m
from physics.ct import *
from utils import show_samples, show_samples_gray, clear, clear_color, batchfy


def grad_op_x(x, w1=1, w2=1): #计算输入图像 x 的梯度
    # Calculate Dx1, Dx2
    Dx1 = (np.roll(x, 1, 2) - x) / 2 * w1
    Dx2 = (np.roll(x, 1, 3) - x) / 2 * w2
    # Ensure Dx1 and Dx2 are 5-dimensional
    Dx1 = Dx1[..., np.newaxis]  # Add an extra dimension
    Dx2 = Dx2[..., np.newaxis]  # Add an extra dimension
    # Concatenate Dx1, Dx2, along the new axis (5th dimension)
    Dx = np.concatenate((Dx1, Dx2), axis=4)
    # Concatenate Dx1, Dx2, along the third axis
    # Dx = np.concatenate((Dx1[..., np.newaxis], Dx2[..., np.newaxis]), axis=2)
    return Dx # 维度从 M*N 变成 M*N*2


def grad_op_x_torch(x, w1=1, w2=1):  # 计算输入图像 x 的梯度
    # 使用 PyTorch 的 roll 函数计算 Dx1 和 Dx2
    Dx1 = (torch.roll(x, shifts=1, dims=2) - x) / 2 * w1  # 在第三个维度上向下滚动
    Dx2 = (torch.roll(x, shifts=1, dims=3) - x) / 2 * w2  # 在第四个维度上向右滚动
    # 在 PyTorch 中，我们直接使用 stack 或 cat 来合并张量
    # 在最后一个维度（通道维度）上堆叠 Dx1 和 Dx2
    Dx = torch.stack((Dx1, Dx2), dim=-1)
    return Dx

    
def adj_grad_op_x(x, w1=1, w2=1): # 此函数计算 grad_op_x 的伴随操作
    y1 = (np.roll(x[..., 0], -1, 2) - x[..., 0]) / 2 * w1
    y2 = (np.roll(x[..., 1], -1, 3) - x[..., 1]) / 2 * w2
    return y1 + y2 # 维度不变
    
def adj_grad_op_x_torch(x, w1=1, w2=1):
    # 计算 grad_op_x 的伴随操作
    y1 = (torch.roll(x[..., 0], shifts=-1, dims=2) - x[..., 0]) / 2 * w1  # 在第三个维度上向上滚动
    y2 = (torch.roll(x[..., 1], shifts=-1, dims=3) - x[..., 1]) / 2 * w2  # 在第四个维度上向左滚动
    return y1 + y2  # 维度不变

    


def grad_op_V(V):
    # 输入 V 的维度为 (448, 1, 256, 256, 2)
    # 计算更复杂的梯度操作
    D1v1 = (np.roll(V[..., 0], shift=1, axis=2) - V[..., 0]) / 2  # 第三维度上的滚动
    D2v2 = (np.roll(V[..., 1], shift=1, axis=3) - V[..., 1]) / 2  # 第四维度上的滚动
    D2v1 = (np.roll(V[..., 0], shift=1, axis=3) - V[..., 0]) / 2  # 第四维度上的滚动
    D1v2 = (np.roll(V[..., 1], shift=1, axis=2) - V[..., 1]) / 2  # 第三维度上的滚动
    # 沿最后一个轴（第五维度）连接结果
    EV = np.concatenate((D1v1[..., np.newaxis], D2v2[..., np.newaxis], (D1v2[..., np.newaxis] + D2v1[..., np.newaxis]) / 2), axis=4)
    return EV
    
def grad_op_V_torch(V):
    # 输入 V 的维度为 (448, 1, 256, 256, 2)
    # 使用 PyTorch 的 roll 函数计算更复杂的梯度操作
    D1v1 = (torch.roll(V[..., 0], shifts=1, dims=2) - V[..., 0]) / 2  # 在第三个维度上向下滚动
    D2v2 = (torch.roll(V[..., 1], shifts=1, dims=3) - V[..., 1]) / 2  # 在第四个维度上向右滚动
    D2v1 = (torch.roll(V[..., 0], shifts=1, dims=3) - V[..., 0]) / 2  # 在第四个维度上向右滚动
    D1v2 = (torch.roll(V[..., 1], shifts=1, dims=2) - V[..., 1]) / 2  # 在第三个维度上向下滚动
    # 在 PyTorch 中，使用 stack 在最后一个维度上堆叠结果
    EV = torch.stack((D1v1, D2v2, (D2v1 + D1v2) / 2), dim=-1)
    return EV
  
  

def adj_grad_op_V(V): #此函数计算 grad_op_V 的伴随操作
    y1 = (np.roll(V[...,0], -1, 0) - V[...,0])/2 + (np.roll(V[...,2], -1, 1) - V[...,2])/2 * 0.5
    y2 = (np.roll(V[...,1], -1, 1) - V[...,1])/2 + (np.roll(V[...,2], -1, 0) - V[...,2])/2 * 0.5
    return np.concatenate((y1[..., np.newaxis], y2[..., np.newaxis]), axis=2)


def adj_grad_op_V_torch(V):
    # 此函数计算 grad_op_V 的伴随操作
    # 使用 PyTorch 的 roll 函数进行滚动操作
    y1 = (torch.roll(V[..., 0], shifts=-1, dims=2) - V[..., 0]) / 2 + (torch.roll(V[..., 2], shifts=-1, dims=3) - V[..., 2]) / 4
    y2 = (torch.roll(V[..., 1], shifts=-1, dims=3) - V[..., 1]) / 2 + (torch.roll(V[..., 2], shifts=-1, dims=2) - V[..., 2]) / 4

    # 在 PyTorch 中，使用 stack 在第四个维度上堆叠结果
    return torch.stack((y1, y2), dim=4)



def get_pc_radon_ADMM_TGV_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lam=0.1,
                             rho_z=10, rho_y=10, alpha_0=1, alpha_1=1):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    # 这两个函数是原函数的部分应用版本，使得后续函数调用，只需要提供未固定的参数即可
    # 所以重点要看 shared_predictor_update_fn 和 shared_corrector_update_fn 两个函数
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)


    X = np.zeros(img_shape)
    print("the shape of X:",X.shape) # the shape of X: (448, 1, 256, 256)
    Dx = grad_op_x(X)
    del_v = Dx
    print("the shape of del_v:",del_v.shape) # the shape of del_v: (448, 1, 256, 256, 2)
    del_z = Dx - del_v
    print("the shape of del_z:",del_z.shape) # the shape of del_z: (448, 1, 256, 256, 2)
    Dv = grad_op_V(del_v)
    del_y = Dv
    print("the shape of del_y:",del_y.shape) # the shape of del_y: (448, 1, 256, 256, 3)
    del_Uz = del_z - Dx + del_v
    print("the shape of del_Uz:",del_Uz.shape) # the shape of del_Uz: (448, 1, 256, 256, 2)
    del_Uy = Dv - del_y
    print("the shape of del_Uy:",del_Uy.shape) # the shape of del_Uy: (448, 1, 256, 256, 3)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lam=1.0, i=None, norm_const=None):
        x = x + lam * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean


    def Z_step(Dx, del_v, del_Uz, del_z, lam=0.1, alpha_1=1, rho_z=10):
        """
        Optimized proximal gradient for l2 norm
        This function is modified to handle inputs with shape (448, 1, 256, 256, 2)
        """
        # 确保 del_v, del_Uz, del_z 与 Dx 的维度一致
        assert Dx.shape == del_v.shape == del_Uz.shape == del_z.shape
        # 展平最后三个维度
        original_shape = Dx.shape
        Dx_flat = Dx.reshape(-1, 256*256*2)
        del_v_flat = del_v.reshape(-1, 256*256*2)
        del_Uz_flat = del_Uz.reshape(-1, 256*256*2)
        # 执行原有的操作
        a = (Dx_flat - del_v_flat - del_Uz_flat).reshape(-1, 2)
        coef = torch.maximum(1 - (2 * lam * alpha_1 / rho_z) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*2)
        # 恢复原始形状
        result = result_flat.reshape(original_shape)
        return result
    
       

    def Y_step(DV, del_Uy, del_y, lam=0.1, alpha_0=1, rho_y=10):
        """
        Optimized proximal gradient for l2 norm
        Modified to handle inputs with shape (448, 1, 256, 256, 3)
        """
        # 确保 del_Uy 和 del_y 与 DV 的维度一致
        assert DV.shape == del_Uy.shape == del_y.shape
        # 展平最后三个维度
        original_shape = DV.shape
        DV_flat = DV.reshape(-1, 256*256*3)
        del_Uy_flat = del_Uy.reshape(-1, 256*256*3)
        # 执行原有的操作
        a = (DV_flat + del_Uy_flat).reshape(-1, 3)
        coef = torch.maximum(1 - (2 * lam * alpha_0 / rho_y) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*3)
        # 恢复原始形状
        result = result_flat.reshape(original_shape)
        return result
    
    
        
    def to_torch_tensor(var, device):
        """
        Convert a variable to a PyTorch tensor if it is a NumPy array.
        If it is already a tensor, just move it to the specified device.
        """
        if isinstance(var, np.ndarray):
            return torch.from_numpy(var).to(device)
        elif isinstance(var, torch.Tensor):
            return var.to(device)
        else:
            raise TypeError("The input must be a numpy array or a PyTorch tensor.")



    def A_cg1(x):
        print("the type of x:",type(x))
        return _AT(_A(x)) + 2 * rho_z * adj_grad_op_x_torch(grad_op_x_torch(x)) #.flatten() # 方程 6

    def A_cg2(v):
        print("the type of v:",type(v))
        return 2 * rho_z * v + 2 * rho_y * adj_grad_op_V_torch(grad_op_V_torch(v)) #.flatten() 方程 9



    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)
            x += a * p
            r -= a * Ap
            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x



    def CS_routine(x, ATy, niter):
        nonlocal del_z, del_v, del_y, del_Uz, del_Uy
        del_v = to_torch_tensor(del_v, x.device)
        del_z = to_torch_tensor(del_z, x.device)
        del_y = to_torch_tensor(del_y, x.device)
        del_Uz = to_torch_tensor(del_Uz, x.device)
        del_Uy = to_torch_tensor(del_Uy, x.device)
        for i in range(niter):
            b_cg1 = ATy + 2 * rho_z * adj_grad_op_x_torch(del_z + del_v + del_Uz) # 方程 7
            x = CG(A_cg1, b_cg1, x, n_inner=1) # 方程 8
            print("del_y data type:",del_y.dtype)
            EtYU = adj_grad_op_V_torch(del_y - del_Uy)
            print("EtYU data type:", EtYU.dtype)
            print("EtYU shape:", EtYU.shape)
            print("grad_op_x_torch(x):",grad_op_x_torch(x).shape)
            b_cg2 = 2 * rho_z * (grad_op_x_torch(x) - del_z - del_Uz) + 2 * rho_y * EtYU # 方程 10
            del_v = CG(A_cg2, b_cg2, del_v, n_inner=1) # 方程 11
            del_z = Z_step(grad_op_x_torch(x),del_v,del_Uz, del_z) #方程 12
            del_y = Y_step(grad_op_V_torch(del_v), del_Uy, del_y) # 方程 13
            del_Uz += del_z - grad_op_x_torch(x) + del_v # 方程 14
            del_Uy += grad_op_V_torch(del_v) - del_y # 方程 15
        x_mean = x
        return x, x_mean
    
    
    

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn



    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1) #方程 6-15
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()


    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            # 传入一个形状和传入参数shape相同的张量，其中元素以标准正态分布采样，且该张量的所有元素都会乘以 ‘self.sigma_max=50'
            ones = torch.ones_like(x).to(data.device) #与张量 x 具有相同形状的张量，其中的所有元素都被设置为 1
            norm_const = _AT(_A(ones)) # 对输入图像先 radon 变换，再逆 radon 变换
            timesteps = torch.linspace(sde.T, eps, sde.N) # linspace(1, 1e-5,1000) 返回包含 1000个元素的张量，元素从 1 到 1e-5的等间隔序列
            for i in tqdm(range(sde.N)): # tqdm用于显示循环进度条
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU 将数据划分成适合 GPU 内存的小批次
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list() # 创建一个空列表
                for idx, x_batch_sing in enumerate(x_batch): #第一个元素是索引值，第二个元素是可迭代对象的元素
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)
                if save_progress:
                    if (i % 20) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lam=1.0, norm_const=norm_const)
            return inverse_scaler(x_mean if denoise else x)
    return pc_radon
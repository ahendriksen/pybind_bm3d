import pybind_bm3d as m
import numpy as np

assert m.__version__ == '0.1.0'

sigma = 1.0
noisy_img = np.random.normal(0, sigma, size=(100, 100)).astype(np.float32)

denoised_img = m.bm3d(noisy_img, sigma)

print(denoised_img.shape)
print(denoised_img.dtype)
print(denoised_img.mean())
print(denoised_img.min(), denoised_img.max())

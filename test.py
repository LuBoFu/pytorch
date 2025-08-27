import torch
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.mm(x, y)
print("GPU 矩阵乘法成功，cuDNN 运行正常！")
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

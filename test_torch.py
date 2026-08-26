import torch
# 查看PyTorch版本
print("PyTorch 版本:", torch.__version__)
# 检查CUDA是否可用
print("CUDA 是否可用:", torch.cuda.is_available())
# 查看可用GPU数量
print("可用GPU数量:", torch.cuda.device_count())
# 如果有GPU，显示当前使用的GPU名称
if torch.cuda.is_available():
   print("当前GPU名称:", torch.cuda.get_device_name(0))
# 查看PyTorch编译时使用的CUDA版本
print("CUDA 版本:", torch.version.cuda)
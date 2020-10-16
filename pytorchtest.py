from __future__ import print_function
import torch

x1 = torch.cuda.is_available()
print('is CUDA available: ' + str(x1))

print('current device ID: ' + str(torch.cuda.current_device()))

print('device ID 0: ' + str(torch.cuda.device(0)))

print('number of CUDA devices: ' + str(torch.cuda.device_count()))

print('cuda device 0 name: ' + str(torch.cuda.get_device_name(0)))
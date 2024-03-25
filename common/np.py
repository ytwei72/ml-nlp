# coding: utf-8
from common.config import GPU

def _axis_list(in_list):
    x = np.arange(len(in_list))
    return x

if GPU:
    import cupy as np
    import cupyx

    device_id = 3
    device = np.cuda.Device(device_id)
    device.use()
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')

    def axis_list(in_list):
        return _axis_list(in_list).tolist()
else:
    import numpy as np

    def axis_list(in_list):
        return _axis_list(in_list)

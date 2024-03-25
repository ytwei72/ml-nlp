#模型下载
from modelscope import snapshot_download


model_dir = snapshot_download('qwen/Qwen1.5-4B')
print(model_dir)


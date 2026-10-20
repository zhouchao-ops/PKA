import torch
import os 
import numpy as np
import functools
GLOBAL_CONFIG = {}
subject_intermediate = {}
 
def extract_mask_from_attn_wlabel(attn: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """
    从注意力图中提取最大连通域作为mask，并放大到1024x1024。
    
    参数:
        attn: 注意力图，形状为 (N,) 或 (H, W)，例如 (32, 32)
        threshold: 二值化阈值 (0-1之间)

    返回:
        mask: 形状为 (1024, 1024) 的 torch.Tensor（值为 0 或 1）
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label, regionprops

    # 转为 numpy 进行图像处理
    device = attn.device
    attn_np = attn.cpu().numpy()
    
    # 归一化到 [0, 1]
    attn_resized = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min())
    
    # 二值化
    binary = (attn_resized > threshold).astype(bool)
    
    # 填充空洞
    filled = binary_fill_holes(binary)
    
    # 标记连通域
    labeled = label(filled)
    
    # 找最大连通域
    max_region = 0
    max_size = 0
    for region in regionprops(labeled):
        if region.area > max_size:
            max_size = region.area
            max_region = region.label
    
    # 构建最大区域的 mask
    if max_region > 0:
        mask = (labeled == max_region).astype(np.float32)
    else:
        mask = np.zeros_like(filled, dtype=np.float32)
    
    # 转换为 torch tensor 并还原设备
    mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    
    return mask_tensor
    

class KVCache:
    def __init__(self) -> None:
        self.key_cache = {}
        self.value_cache = {}
    
    def __getitem__(self, layer_idx: str) -> tuple[torch.Tensor,torch.Tensor]:
        assert layer_idx in self.key_cache 
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __iter__(self):
        for layer_idx in self.key_cache:
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        layer_idx,
        key_states,
        value_states,
    ):
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states

import torch
import os
import functools

def conditional_gpu_profile(func=None, device=0):
    """
    一个条件式的GPU显存分析装饰器。

    它会精确计算函数执行期间，相对于函数开始时的“峰值显存增量”。
    仅当环境变量 `PROFILE_GPU_MEMORY` 设置为 '1' 或 'true' (不区分大小写) 时激活。
    """
    if func is None:
        return functools.partial(conditional_gpu_profile, device=device)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profile_env = os.environ.get('PROFILE_GPU_MEMORY', 'false').lower()
        should_profile = profile_env in ['1', 'true']

        if not should_profile:
            return func(*args, **kwargs)

        if not torch.cuda.is_available():
            print(f"警告: 函数 '{func.__name__}' 要求进行GPU分析，但CUDA不可用。")
            return func(*args, **kwargs)

        # --- 核心分析逻辑 ---
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        
        # (A) 执行前
        mem_before = torch.cuda.memory_allocated(device=device)
        
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize(device=device)
        # (B) 执行后
        mem_after = torch.cuda.memory_allocated(device=device)
        # (P) 执行期间的峰值
        peak_mem = torch.cuda.max_memory_allocated(device=device)
        
        # --- 显存报告 ---
        peak_increase = peak_mem - mem_before
        net_increase = mem_after - mem_before

        print(f"--- GPU显存分析报告 for '{func.__name__}' (Device: cuda:{device}) ---")
        print(f"  (A) 执行前已分配: {mem_before / 1024**2:,.3f} MB")
        print(f"  (B) 执行后已分配: {mem_after / 1024**2:,.3f} MB")
        print(f"  (P) 执行中峰值:   {peak_mem / 1024**2:,.3f} MB")
        print("--------------------------------------------------")
        # 这个指标反映了函数返回后，显存的永久性增加
        print(f"  函数净增显存 (B - A): {net_increase / 1024**2:,.3f} MB")
        # 这个指标反映了函数为了完成计算，所需要的最大额外空间
        print(f"  函数峰值增量 (P - A): {peak_increase / 1024**2:,.3f} MB  <-- [关键指标]")
        print(f"--------------------------------------------------")
        
        return result

    return wrapper

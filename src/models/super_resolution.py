"""
Super Resolution Models
超解析度模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Union, Tuple

class SuperResolution:
    """超分辨率模型
    
    使用簡單的雙三次插值進行超分辨率處理
    """
    
    def __init__(self, scale_factor: int = 2):
        """初始化超分辨率模型
        
        Args:
            scale_factor: 放大倍數，默認為2
        """
        self.scale_factor = scale_factor
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """處理圖像
        
        Args:
            image: 輸入圖像，格式為 numpy.ndarray
            
        Returns:
            處理後的圖像
        """
        # 確保圖像是 BGR 格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # 使用雙三次插值進行放大
        height, width = image.shape[:2]
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor
        
        # 使用雙三次插值進行放大
        upscaled = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        return upscaled

class SRCNN(nn.Module):
    """SRCNN模型架構"""
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x 
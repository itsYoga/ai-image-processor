"""
Auto White Balance implementation
自動白平衡實現
"""

import cv2
import numpy as np

class AutoWhiteBalance:
    """
    Auto White Balance class for automatic white balance control
    自動白平衡控制類
    """
    
    def __init__(self):
        """
        Initialize Auto White Balance parameters
        初始化自動白平衡參數
        """
        self.method = 'gray_world'  # White balance method 白平衡方法
        self.max_gain = 2.0         # Maximum gain value 最大增益值
        self.min_gain = 0.5         # Minimum gain value 最小增益值
    
    def gray_world(self, image):
        """
        Apply Gray World white balance algorithm
        應用灰度世界白平衡演算法
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            White balanced image 白平衡後的影像
        """
        # Convert to float for processing
        # 轉換為浮點數進行處理
        img_float = image.astype(np.float32)
        
        # Calculate average values for each channel
        # 計算每個通道的平均值
        avg_r = np.mean(img_float[:,:,0])
        avg_g = np.mean(img_float[:,:,1])
        avg_b = np.mean(img_float[:,:,2])
        
        # Calculate scaling factors
        # 計算縮放因子
        scale_r = avg_g / avg_r if avg_r > 0 else 1.0
        scale_b = avg_g / avg_b if avg_b > 0 else 1.0
        
        # Clamp scaling factors
        # 限制縮放因子
        scale_r = np.clip(scale_r, self.min_gain, self.max_gain)
        scale_b = np.clip(scale_b, self.min_gain, self.max_gain)
        
        # Apply white balance
        # 應用白平衡
        img_float[:,:,0] *= scale_r
        img_float[:,:,2] *= scale_b
        
        # Convert back to uint8
        # 轉換回uint8
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    def perfect_reflector(self, image):
        """
        Apply Perfect Reflector white balance algorithm
        應用完美反射體白平衡演算法
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            White balanced image 白平衡後的影像
        """
        # Convert to float for processing
        # 轉換為浮點數進行處理
        img_float = image.astype(np.float32)
        
        # Find the brightest pixels
        # 找到最亮的像素
        brightness = np.mean(img_float, axis=2)
        threshold = np.percentile(brightness, 99)
        bright_pixels = brightness > threshold
        
        # Calculate scaling factors from bright pixels
        # 從亮像素計算縮放因子
        avg_r = np.mean(img_float[bright_pixels, 0])
        avg_g = np.mean(img_float[bright_pixels, 1])
        avg_b = np.mean(img_float[bright_pixels, 2])
        
        # Calculate scaling factors
        # 計算縮放因子
        scale_r = avg_g / avg_r if avg_r > 0 else 1.0
        scale_b = avg_g / avg_b if avg_b > 0 else 1.0
        
        # Clamp scaling factors
        # 限制縮放因子
        scale_r = np.clip(scale_r, self.min_gain, self.max_gain)
        scale_b = np.clip(scale_b, self.min_gain, self.max_gain)
        
        # Apply white balance
        # 應用白平衡
        img_float[:,:,0] *= scale_r
        img_float[:,:,2] *= scale_b
        
        # Convert back to uint8
        # 轉換回uint8
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    def process_image(self, image):
        """
        Process an image with auto white balance
        使用自動白平衡處理影像
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            White balanced image 白平衡後的影像
        """
        if self.method == 'gray_world':
            return self.gray_world(image)
        elif self.method == 'perfect_reflector':
            return self.perfect_reflector(image)
        else:
            return image
            
    def set_method(self, method):
        """
        Set the white balance method
        設置白平衡方法
        
        Args:
            method: White balance method ('gray_world' or 'perfect_reflector')
                   白平衡方法（'gray_world'或'perfect_reflector'）
        """
        if method not in ['gray_world', 'perfect_reflector']:
            raise ValueError("Method must be either 'gray_world' or 'perfect_reflector'")
        self.method = method 
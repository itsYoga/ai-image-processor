"""
Auto Exposure implementation
自動曝光實現
"""

import cv2
import numpy as np

class AutoExposure:
    """
    Auto Exposure class for automatic exposure control
    自動曝光控制類
    """
    
    def __init__(self):
        """
        Initialize Auto Exposure parameters
        初始化自動曝光參數
        """
        self.target_brightness = 0.5  # Target brightness level (0-1) 目標亮度水平（0-1）
        self.max_adjustment = 2.0  # Maximum exposure adjustment factor 最大曝光調整因子
        self.min_adjustment = 0.5  # Minimum exposure adjustment factor 最小曝光調整因子
    
    def calculate_brightness(self, image):
        """
        Calculate the average brightness of an image
        計算影像的平均亮度
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            Average brightness (0-1) 平均亮度（0-1）
        """
        # Convert to grayscale if needed
        # 如果需要，轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate average brightness
        # 計算平均亮度
        return np.mean(gray) / 255.0
    
    def calculate_exposure_adjustment(self, current_brightness):
        """
        Calculate the required exposure adjustment factor
        計算所需的曝光調整因子
        
        Args:
            current_brightness: Current image brightness (0-1) 當前影像亮度（0-1）
            
        Returns:
            Exposure adjustment factor 曝光調整因子
        """
        # Calculate adjustment factor
        # 計算調整因子
        if current_brightness == 0:
            return self.max_adjustment
            
        adjustment = self.target_brightness / current_brightness
        
        # Clamp adjustment factor
        # 限制調整因子
        return np.clip(adjustment, self.min_adjustment, self.max_adjustment)
    
    def process_image(self, image):
        """
        Process an image with auto exposure
        使用自動曝光處理影像
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            Tuple of (processed image, exposure adjustment factor)
            元組（處理後的影像，曝光調整因子）
        """
        # Calculate current brightness
        # 計算當前亮度
        current_brightness = self.calculate_brightness(image)
        
        # Calculate exposure adjustment
        # 計算曝光調整
        adjustment = self.calculate_exposure_adjustment(current_brightness)
        
        # Apply exposure adjustment
        # 應用曝光調整
        adjusted_image = cv2.convertScaleAbs(image, alpha=adjustment, beta=0)
        
        return adjusted_image, adjustment 
"""
Auto Focus implementation
自動對焦實現
"""

import cv2
import numpy as np

class AutoFocus:
    """
    Auto Focus class for automatic focus control
    自動對焦控制類
    """
    
    def __init__(self):
        """
        Initialize Auto Focus parameters
        初始化自動對焦參數
        """
        self.max_focus_pos = 100  # Maximum focus position 最大對焦位置
        self.min_focus_pos = 0    # Minimum focus position 最小對焦位置
        self.focus_step = 1       # Focus adjustment step 對焦調整步長
        self.current_pos = 50     # Current focus position 當前對焦位置
    
    def calculate_focus_measure(self, image):
        """
        Calculate focus measure using Laplacian variance
        使用拉普拉斯方差計算對焦度量
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            Focus measure value 對焦度量值
        """
        # Convert to grayscale if needed
        # 如果需要，轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate Laplacian variance
        # 計算拉普拉斯方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def find_best_focus(self, image):
        """
        Find the best focus position by scanning through focus range
        通過掃描對焦範圍找到最佳對焦位置
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            Tuple of (best focus position, best focus measure)
            元組（最佳對焦位置，最佳對焦度量）
        """
        best_measure = 0
        best_pos = self.current_pos
        
        # Scan through focus range
        # 掃描對焦範圍
        for pos in range(self.min_focus_pos, self.max_focus_pos + 1, self.focus_step):
            # Simulate focus adjustment (in real camera, this would control the lens)
            # 模擬對焦調整（在實際相機中，這將控制鏡頭）
            self.current_pos = pos
            
            # Calculate focus measure at current position
            # 計算當前位置的對焦度量
            measure = self.calculate_focus_measure(image)
            
            # Update best position if current measure is better
            # 如果當前度量更好，更新最佳位置
            if measure > best_measure:
                best_measure = measure
                best_pos = pos
        
        return best_pos, best_measure
    
    def process_image(self, image):
        """
        Process an image with auto focus
        使用自動對焦處理影像
        
        Args:
            image: Input image 輸入影像
            
        Returns:
            Tuple of (processed image, focus position, focus measure)
            元組（處理後的影像，對焦位置，對焦度量）
        """
        # Find best focus position
        # 找到最佳對焦位置
        focus_pos, focus_measure = self.find_best_focus(image)
        
        # Apply focus adjustment (in real camera, this would control the lens)
        # 應用對焦調整（在實際相機中，這將控制鏡頭）
        # For simulation, we'll just return the original image
        # 對於模擬，我們只返回原始影像
        return image, focus_pos, focus_measure 
"""
Image Processing Module
影像處理模組
"""

import cv2
import numpy as np
from skimage import feature, filters, morphology

class ImageProcessor:
    """
    Image Processing class that combines all image processing features
    整合所有影像處理功能的類
    """
    
    def __init__(self):
        """
        Initialize image processing components
        初始化影像處理組件
        """
        self.auto_exposure = AutoExposure()
        self.auto_focus = AutoFocus()
        self.auto_wb = AutoWhiteBalance()
        self.vision_algs = VisionAlgorithms()
        self.isp_pipeline = ISPPipeline()
    
    def process_image(self, image, operations=None):
        """
        Process image with specified operations
        使用指定的操作處理影像
        
        Args:
            image: Input image 輸入影像
            operations: List of operations to apply 要應用的操作列表
            
        Returns:
            Processed image and processing info 處理後的影像和處理信息
        """
        if operations is None:
            operations = ['auto_exposure', 'auto_wb', 'noise_reduction']
            
        result = image.copy()
        info = {}
        
        for op in operations:
            if op == 'auto_exposure':
                result, exposure_factor = self.auto_exposure.process_image(result)
                info['exposure_factor'] = exposure_factor
                
            elif op == 'auto_focus':
                result, focus_pos, focus_measure = self.auto_focus.process_image(result)
                info['focus_position'] = focus_pos
                info['focus_measure'] = focus_measure
                
            elif op == 'auto_wb':
                result = self.auto_wb.process_image(result)
                
            elif op == 'edge_detection':
                result = self.vision_algs.detect_edges(result)
                
            elif op == 'feature_detection':
                result, keypoints = self.vision_algs.detect_features(result)
                info['keypoints'] = len(keypoints)
                
            elif op == 'enhancement':
                result = self.vision_algs.enhance_image(result)
                
            elif op == 'noise_reduction':
                result = self.vision_algs.reduce_noise(result)
                
            elif op == 'segmentation':
                result = self.vision_algs.segment_image(result)
                
        return result, info

# Import all the image processing classes
from .auto_exposure import AutoExposure
from .auto_focus import AutoFocus
from .auto_white_balance import AutoWhiteBalance
from .vision_algorithms import VisionAlgorithms
from .isp_pipeline import ISPPipeline 
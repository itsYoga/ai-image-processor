import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    def __init__(self):
        self.gamma = 1.0
        self.contrast = 1.0
        self.brightness = 0
        self.saturation = 1.0

    def adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """調整圖像的伽馬值"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def adjust_contrast_brightness(self, image: np.ndarray, contrast: float, brightness: int) -> np.ndarray:
        """調整圖像的對比度和亮度"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        """調整圖像的飽和度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """自動白平衡"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """降噪處理"""
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """銳化處理"""
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def process_image(self, image: np.ndarray, 
                     gamma: Optional[float] = None,
                     contrast: Optional[float] = None,
                     brightness: Optional[int] = None,
                     saturation: Optional[float] = None,
                     auto_wb: bool = False,
                     denoise: bool = False,
                     sharpen: bool = False) -> np.ndarray:
        """整合所有影像處理功能"""
        if gamma is not None:
            image = self.adjust_gamma(image, gamma)
        if contrast is not None or brightness is not None:
            image = self.adjust_contrast_brightness(
                image, 
                contrast if contrast is not None else self.contrast,
                brightness if brightness is not None else self.brightness
            )
        if saturation is not None:
            image = self.adjust_saturation(image, saturation)
        if auto_wb:
            image = self.auto_white_balance(image)
        if denoise:
            image = self.denoise(image)
        if sharpen:
            image = self.sharpen(image)
        return image 
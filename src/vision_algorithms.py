"""
Computer Vision Algorithms implementation
電腦視覺演算法實現
"""

import cv2
import numpy as np
from skimage import feature, filters, morphology

class VisionAlgorithms:
    """
    Computer Vision Algorithms class
    電腦視覺演算法類
    """
    
    def __init__(self):
        """
        Initialize Vision Algorithms parameters
        初始化視覺演算法參數
        """
        self.edge_methods = ['canny', 'sobel']  # Available edge detection methods 可用的邊緣檢測方法
        self.feature_methods = ['sift', 'orb']  # Available feature detection methods 可用的特徵檢測方法
        self.enhancement_methods = ['clahe', 'histogram']  # Available enhancement methods 可用的增強方法
        self.noise_reduction_methods = ['nlm', 'bilateral']  # Available noise reduction methods 可用的降噪方法
        self.segmentation_methods = ['watershed', 'kmeans']  # Available segmentation methods 可用的分割方法
    
    def detect_edges(self, image, method='canny', low_threshold=50, high_threshold=150):
        """
        Detect edges in an image
        檢測影像中的邊緣
        
        Args:
            image: Input image 輸入影像
            method: Edge detection method ('canny' or 'sobel') 邊緣檢測方法（'canny'或'sobel'）
            low_threshold: Lower threshold for Canny edge detection 對Canny邊緣檢測的低閾值
            high_threshold: Upper threshold for Canny edge detection 對Canny邊緣檢測的高閾值
            
        Returns:
            Edge detected image 邊緣檢測後的影像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if method == 'canny':
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges * 255 / edges.max()).astype(np.uint8)
        elif method == 'prewitt':
            edges = filters.prewitt(gray)
            edges = (edges * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
            
        return edges
        
    def detect_features(self, image, method='sift', max_features=100):
        """
        Detect features in an image
        檢測影像中的特徵
        
        Args:
            image: Input image 輸入影像
            method: Feature detection method ('sift' or 'orb') 特徵檢測方法（'sift'或'orb'）
            max_features: Maximum number of features to detect 要檢測的最大特徵數
            
        Returns:
            Tuple of (feature image, keypoints) 元組（特徵影像，關鍵點）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if method == 'sift':
            detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == 'orb':
            detector = cv2.ORB_create(nfeatures=max_features)
        elif method == 'fast':
            detector = cv2.FastFeatureDetector_create()
        else:
            raise ValueError(f"Unknown feature detection method: {method}")
            
        keypoints = detector.detect(gray, None)
        
        # Draw keypoints on image
        result = cv2.drawKeypoints(image, keypoints, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return result, keypoints
        
    def enhance_image(self, image, method='clahe'):
        """
        Enhance image quality
        增強影像品質
        
        Args:
            image: Input image 輸入影像
            method: Enhancement method ('clahe' or 'histogram') 增強方法（'clahe'或'histogram'）
            
        Returns:
            Enhanced image 增強後的影像
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image
            
        if method == 'clahe':
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_l = clahe.apply(l)
        elif method == 'histogram_equalization':
            enhanced_l = cv2.equalizeHist(l)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
            
        if len(image.shape) == 3:
            # Merge channels back
            enhanced_lab = cv2.merge([enhanced_l, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            result = enhanced_l
            
        return result
        
    def reduce_noise(self, image, method='nlm', strength=10):
        """
        Reduce noise in an image
        降低影像中的雜訊
        
        Args:
            image: Input image 輸入影像
            method: Noise reduction method ('nlm' or 'bilateral') 降噪方法（'nlm'或'bilateral'）
            strength: Strength parameter for Non-Local Means denoising 非局部均值降噪的強度參數
            
        Returns:
            Denoised image 降噪後的影像
        """
        if method == 'nlm':
            # Non-local means denoising
            result = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        elif method == 'bilateral':
            # Bilateral filter
            result = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            # Gaussian blur
            result = cv2.GaussianBlur(image, (5, 5), 0)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
            
        return result
        
    def segment_image(self, image, method='watershed'):
        """
        Segment an image
        分割影像
        
        Args:
            image: Input image 輸入影像
            method: Segmentation method ('watershed' or 'kmeans') 分割方法（'watershed'或'kmeans'）
            
        Returns:
            Segmented image 分割後的影像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if method == 'watershed':
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown==255] = 0
            
            # Apply watershed
            markers = cv2.watershed(image, markers)
            result = markers
            
        elif method == 'kmeans':
            # Reshape image
            pixel_values = image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Define criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8
            centers = np.uint8(centers)
            result = centers[labels.flatten()]
            result = result.reshape(image.shape)
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
            
        return result 
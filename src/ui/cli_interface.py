"""
Command Line Interface for Image Processing
影像處理的命令行界面
"""

import argparse
import cv2
import os
from src.image_processing import ImageProcessor

class CLInterface:
    """
    Command Line Interface class
    命令行界面類
    """
    
    def __init__(self):
        """
        Initialize the interface
        初始化界面
        """
        self.processor = ImageProcessor()
        
    def parse_args(self):
        """
        Parse command line arguments
        解析命令行參數
        
        Returns:
            Parsed arguments 解析後的參數
        """
        parser = argparse.ArgumentParser(description="Image Processing CLI")
        
        parser.add_argument("input_path", help="Path to input image")
        parser.add_argument("output_path", help="Path to save processed image")
        
        parser.add_argument("--operations", nargs="+", choices=[
            "auto_exposure",
            "auto_wb",
            "edge_detection",
            "feature_detection",
            "enhancement",
            "noise_reduction",
            "segmentation"
        ], help="Operations to apply")
        
        parser.add_argument("--edge-method", choices=["canny", "sobel", "prewitt"],
                          default="canny", help="Edge detection method")
        
        parser.add_argument("--feature-method", choices=["sift", "orb", "fast"],
                          default="sift", help="Feature detection method")
        
        parser.add_argument("--enhancement-method", choices=["clahe", "histogram"],
                          default="clahe", help="Enhancement method")
        
        parser.add_argument("--noise-method", choices=["nlm", "bilateral", "gaussian"],
                          default="nlm", help="Noise reduction method")
        
        parser.add_argument("--segmentation-method", choices=["watershed", "kmeans"],
                          default="watershed", help="Segmentation method")
        
        return parser.parse_args()
    
    def run(self):
        """
        Run the CLI
        運行命令行界面
        """
        args = self.parse_args()
        
        # Check if input file exists
        if not os.path.exists(args.input_path):
            print(f"Error: Input file '{args.input_path}' does not exist")
            return
        
        # Read input image
        image = cv2.imread(args.input_path)
        if image is None:
            print(f"Error: Could not read image '{args.input_path}'")
            return
        
        # Process image
        result, info = self.processor.process_image(image, args.operations)
        
        # Save result
        cv2.imwrite(args.output_path, result)
        
        # Print processing information
        print("\nProcessing Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print(f"\nProcessed image saved to: {args.output_path}")

if __name__ == "__main__":
    cli = CLInterface()
    cli.run() 
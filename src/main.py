"""
Main entry point for the Image Processing application
影像處理應用程序的主入口點
"""

import argparse
from ui.image_processing_interface import ImageProcessingInterface
from ui.cli_interface import CLInterface

def parse_args():
    """
    Parse command line arguments
    解析命令行參數
    
    Returns:
        Parsed arguments 解析後的參數
    """
    parser = argparse.ArgumentParser(description="Image Processing Application")
    
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui",
                      help="Interface mode (gui or cli)")
    
    parser.add_argument("--share", action="store_true",
                      help="Create a public link for the Gradio interface")
    
    return parser.parse_args()

def main():
    """
    Main function
    主函數
    """
    args = parse_args()
    
    if args.mode == "gui":
        # Launch Gradio interface
        interface = ImageProcessingInterface()
        interface.launch(share=args.share)
    else:
        # Launch CLI
        cli = CLInterface()
        cli.run()

if __name__ == "__main__":
    main() 
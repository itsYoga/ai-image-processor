"""
Image Processing Interface
影像處理界面
"""

import gradio as gr
import cv2
import numpy as np
from src.image_processing import ImageProcessor
from src.llm_interface.gemini_interface import GeminiInterface
import os
from dotenv import load_dotenv

class ImageProcessingInterface:
    """
    Image Processing Interface class
    影像處理界面類
    """
    
    def __init__(self):
        """
        Initialize the interface
        初始化界面
        """
        self.processor = ImageProcessor()
        # 載入環境變數
        load_dotenv()
        # 初始化 Gemini 介面
        self.gemini = GeminiInterface(os.getenv('GEMINI_API_KEY'))
        
    def create_interface(self):
        """
        Create the Gradio interface
        創建Gradio界面
        
        Returns:
            Gradio interface Gradio界面
        """
        with gr.Blocks(title="影像處理") as interface:
            gr.Markdown("# 影像處理介面")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="輸入圖片")
                    
                    with gr.Group():
                        gr.Markdown("### 處理方式")
                        use_natural_language = gr.Checkbox(
                            label="使用自然語言指令",
                            value=False
                        )
                        
                        natural_language_input = gr.Textbox(
                            label="自然語言指令",
                            placeholder="請用中文描述想要的影像處理，例如：'幫我去除雜訊並自動白平衡'",
                            visible=False
                        )
                        
                        gr.Markdown("### 手動處理選項")
                        operations = gr.CheckboxGroup(
                            choices=[
                                "自動曝光",
                                "自動白平衡",
                                "邊緣偵測",
                                "特徵點偵測",
                                "增強處理",
                                "降噪處理",
                                "影像分割"
                            ],
                            label="選擇處理項目"
                        )
                        
                        edge_method = gr.Dropdown(
                            choices=["Canny", "Sobel", "Prewitt"],
                            value="Canny",
                            label="邊緣偵測方法"
                        )
                        
                        feature_method = gr.Dropdown(
                            choices=["SIFT", "ORB", "FAST"],
                            value="SIFT",
                            label="特徵點偵測方法"
                        )
                        
                        enhancement_method = gr.Dropdown(
                            choices=["CLAHE", "直方圖均衡"],
                            value="CLAHE",
                            label="增強方法"
                        )
                        
                        noise_method = gr.Dropdown(
                            choices=["NLM非區域平均", "雙邊濾波", "高斯濾波"],
                            value="NLM非區域平均",
                            label="降噪方法"
                        )
                        
                        segmentation_method = gr.Dropdown(
                            choices=["分水嶺", "K-means"],
                            value="分水嶺",
                            label="分割方法"
                        )
                    
                    process_btn = gr.Button("開始處理")
                
                with gr.Column():
                    output_image = gr.Image(label="處理後圖片")
                    info_text = gr.Textbox(label="處理資訊")
            
            def toggle_natural_language(use_nl):
                return gr.update(visible=use_nl)
            
            def process_image(image, use_nl, nl_input, ops, edge_m, feature_m, enhance_m, noise_m, seg_m):
                if image is None:
                    return None, "請上傳圖片"
                
                # 將操作轉換為內部格式
                operations = []
                selected_ops = []
                
                if use_nl and nl_input.strip():
                    # 使用自然語言指令
                    try:
                        nl_operations = self.gemini.parse_instruction(nl_input)
                        if nl_operations:
                            # 將 Gemini 解析結果轉換為操作列表
                            if nl_operations.get("auto_wb"):
                                operations.append("auto_wb")
                                selected_ops.append("自動白平衡")
                            if nl_operations.get("denoise"):
                                operations.append("noise_reduction")
                                selected_ops.append("降噪處理")
                            if nl_operations.get("sharpen"):
                                operations.append("enhancement")
                                selected_ops.append("增強處理")
                            if nl_operations.get("gamma") or nl_operations.get("contrast") or nl_operations.get("brightness"):
                                operations.append("auto_exposure")
                                selected_ops.append("自動曝光")
                    except Exception as e:
                        return None, f"自然語言解析錯誤：{str(e)}"
                
                # 如果沒有自然語言指令或解析失敗，使用手動選項
                if not operations:
                    if "自動曝光" in ops:
                        operations.append("auto_exposure")
                    if "自動白平衡" in ops:
                        operations.append("auto_wb")
                    if "邊緣偵測" in ops:
                        operations.append("edge_detection")
                    if "特徵點偵測" in ops:
                        operations.append("feature_detection")
                    if "增強處理" in ops:
                        operations.append("enhancement")
                    if "降噪處理" in ops:
                        operations.append("noise_reduction")
                    if "影像分割" in ops:
                        operations.append("segmentation")
                else:
                    # 更新手動選項的選擇
                    ops = selected_ops
                
                # 處理圖片
                result, info = self.processor.process_image(image, operations)
                
                # 格式化資訊文字
                info_text = "處理資訊：\n"
                for key, value in info.items():
                    info_text += f"{key}: {value}\n"
                
                return result, info_text, gr.update(value=ops)
            
            use_natural_language.change(
                fn=toggle_natural_language,
                inputs=[use_natural_language],
                outputs=[natural_language_input]
            )
            
            process_btn.click(
                fn=process_image,
                inputs=[
                    input_image,
                    use_natural_language,
                    natural_language_input,
                    operations,
                    edge_method,
                    feature_method,
                    enhancement_method,
                    noise_method,
                    segmentation_method
                ],
                outputs=[output_image, info_text, operations]
            )
        
        return interface
    
    def launch(self, share=False):
        """
        Launch the interface
        啟動界面
        
        Args:
            share: Whether to create a public link 是否創建公共鏈接
        """
        interface = self.create_interface()
        interface.launch(share=share)

if __name__ == "__main__":
    # Create and launch the interface
    interface = ImageProcessingInterface()
    interface.launch() 
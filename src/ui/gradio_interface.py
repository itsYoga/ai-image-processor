import gradio as gr
import cv2
import numpy as np
from typing import Tuple, Optional
import os
from src.image_processing.isp import ImageProcessor
from src.llm_interface.gemini_interface import GeminiInterface

class GradioInterface:
    def __init__(self, gemini_api_key: str):
        """初始化Gradio介面"""
        self.image_processor = ImageProcessor()
        self.llm_interface = GeminiInterface(gemini_api_key)
        
    def process_image_with_instruction(self, 
                                     image: np.ndarray, 
                                     instruction: str) -> Tuple[np.ndarray, str]:
        """根據指令處理圖像"""
        if image is None:
            return None, "請先上傳圖片"
        
        # 解析指令
        operations = self.llm_interface.parse_instruction(instruction)
        if not operations:
            return image, "無法理解指令，請重新輸入"
        
        # 執行圖像處理
        processed_image = self.image_processor.process_image(
            image,
            gamma=operations.get("gamma"),
            contrast=operations.get("contrast"),
            brightness=operations.get("brightness"),
            saturation=operations.get("saturation"),
            auto_wb=operations.get("auto_wb", False),
            denoise=operations.get("denoise", False),
            sharpen=operations.get("sharpen", False)
        )
        
        # 生成處理說明
        description = "已執行以下操作：\n"
        for op, value in operations.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        description += f"- {self.llm_interface.get_operation_description(op)}\n"
                else:
                    description += f"- {self.llm_interface.get_operation_description(op)}: {value}\n"
        
        return processed_image, description

    def create_interface(self) -> gr.Interface:
        """創建Gradio介面"""
        return gr.Interface(
            fn=self.process_image_with_instruction,
            inputs=[
                gr.Image(label="上傳圖片", type="numpy"),
                gr.Textbox(label="處理指令", placeholder="請輸入處理指令，例如：'幫我把這張照片的雜訊去掉，顏色調自然一點'")
            ],
            outputs=[
                gr.Image(label="處理結果"),
                gr.Textbox(label="處理說明")
            ],
            title="智慧型圖像優化系統",
            description="""
            這是一個整合了深度學習和傳統圖像處理技術的智慧型圖像優化系統。
            您可以上傳圖片並使用自然語言描述您想要的處理效果。
            系統支援以下功能：
            - 自動白平衡
            - 降噪處理
            - 銳化處理
            - 亮度/對比度調整
            - 飽和度調整
            - 伽馬值調整
            """,
            examples=[
                ["example_images/noisy.jpg", "幫我把這張照片的雜訊去掉，顏色調自然一點"],
                ["example_images/dark.jpg", "讓這張照片更亮，對比度高一些"],
                ["example_images/blur.jpg", "讓這張照片更清晰，增加一點飽和度"]
            ]
        ) 
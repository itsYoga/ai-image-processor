import gradio as gr
import cv2
import numpy as np
from typing import Tuple, Optional
import os
from src.image_processing.isp import ImageProcessor
from src.llm_interface.gemini_interface import GeminiInterface
from src.models.denoising import Denoising
from src.models.super_resolution import SuperResolution
from src.models.style_transfer import StyleTransfer
from src.models.image_restoration import ImageRestoration

class GradioInterface:
    def __init__(self, gemini_api_key: str):
        """初始化Gradio介面"""
        self.image_processor = ImageProcessor()
        self.llm_interface = GeminiInterface(gemini_api_key)
        
        # 初始化深度學習模型
        self.denoiser = Denoising(model_type='dncnn')
        self.sr_model = SuperResolution()
        self.style_transfer = StyleTransfer()
        self.restoration = ImageRestoration(model_type='unet')
        
    def process_image(self, image: np.ndarray, operation: str) -> Tuple[np.ndarray, str]:
        """根據用戶選擇的操作處理圖像"""
        if image is None:
            return None, "請先上傳圖片"
        processed_image = image.copy()
        description = "已執行操作："
        if operation == "超分辨率處理":
            processed_image = self.sr_model.process(processed_image)
            description += "超分辨率處理"
        elif operation == "風格遷移":
            processed_image = self.style_transfer.process(processed_image)
            description += "風格遷移"
        elif operation == "圖像修復":
            processed_image = self.restoration.process(processed_image)
            description += "圖像修復"
        elif operation == "深度學習降噪":
            processed_image = self.denoiser.process(processed_image)
            description += "深度學習降噪"
        elif operation == "自動白平衡":
            processed_image = self.image_processor.process_image(processed_image, auto_wb=True)
            description += "自動白平衡"
        elif operation == "降噪處理":
            processed_image = self.image_processor.process_image(processed_image, denoise=True)
            description += "傳統降噪"
        elif operation == "銳化處理":
            processed_image = self.image_processor.process_image(processed_image, sharpen=True)
            description += "銳化處理"
        elif operation == "亮度調整":
            processed_image = self.image_processor.process_image(processed_image, brightness=20)
            description += "亮度調整"
        elif operation == "對比度調整":
            processed_image = self.image_processor.process_image(processed_image, contrast=1.2)
            description += "對比度調整"
        elif operation == "飽和度調整":
            processed_image = self.image_processor.process_image(processed_image, saturation=1.2)
            description += "飽和度調整"
        elif operation == "伽馬值調整":
            processed_image = self.image_processor.process_image(processed_image, gamma=1.2)
            description += "伽馬值調整"
        else:
            description = "未選擇操作或操作無效"
        return processed_image, description

    def create_interface(self) -> gr.Interface:
        """創建Gradio介面（手動選擇操作，不依賴LLM）"""
        return gr.Interface(
            fn=self.process_image,
            inputs=[
                gr.Image(label="上傳圖片", type="numpy"),
                gr.Dropdown(
                    label="選擇影像處理操作",
                    choices=[
                        "超分辨率處理",
                        "風格遷移",
                        "圖像修復",
                        "深度學習降噪",
                        "自動白平衡",
                        "降噪處理",
                        "銳化處理",
                        "亮度調整",
                        "對比度調整",
                        "飽和度調整",
                        "伽馬值調整"
                    ],
                    value="超分辨率處理"
                )
            ],
            outputs=[
                gr.Image(label="處理結果"),
                gr.Textbox(label="處理說明")
            ],
            title="智慧型圖像優化系統（手動模式）",
            description="""
            這是一個整合了深度學習和傳統圖像處理技術的智慧型圖像優化系統。
            您可以上傳圖片並手動選擇要執行的影像處理操作。
            支援功能：
            - 超分辨率處理
            - 風格遷移
            - 圖像修復
            - 深度學習降噪
            - 自動白平衡
            - 傳統降噪
            - 銳化處理
            - 亮度/對比度/飽和度/伽馬值調整
            """
        ) 
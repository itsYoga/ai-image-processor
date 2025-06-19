import google.generativeai as genai
from typing import Dict, Optional
import os
import json
import time
from google.api_core import retry
import logging

class GeminiInterface:
    def __init__(self, api_key: str):
        """初始化 Gemini 介面"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # 操作描述映射
        self.operation_descriptions = {
            "gamma": "伽馬值調整",
            "contrast": "對比度調整",
            "brightness": "亮度調整",
            "saturation": "飽和度調整",
            "auto_wb": "自動白平衡",
            "denoise": "降噪處理",
            "sharpen": "銳化處理",
            "super_resolution": "超分辨率處理",
            "style_transfer": "風格遷移",
            "restoration": "圖像修復"
        }
        
        # 設置日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @retry.Retry(
        predicate=retry.if_exception_type(Exception),
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=30.0
    )
    def _call_gemini_api(self, prompt: str) -> str:
        """呼叫 Gemini API 並處理重試邏輯"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            return response.text
        except Exception as e:
            self.logger.error(f"API 呼叫錯誤: {str(e)}")
            if "quota" in str(e).lower():
                time.sleep(5)  # 減少等待時間
            raise

    def parse_instruction(self, instruction: str) -> Optional[Dict]:
        """解析自然語言指令"""
        prompt = f"""
        請分析以下圖像處理指令，並返回相應的處理參數。
        指令：{instruction}
        
        請返回一個字典，包含以下可能的參數：
        - gamma: 伽馬值（0.5-2.0）
        - contrast: 對比度（0.5-1.5）
        - brightness: 亮度（-50到50）
        - saturation: 飽和度（0.5-1.5）
        - auto_wb: 是否自動白平衡（True/False）
        - denoise: 是否降噪（True/False）
        - sharpen: 是否銳化（True/False）
        - super_resolution: 是否進行超分辨率處理（True/False）
        - style_transfer: 是否進行風格遷移（True/False）
        - restoration: 是否進行圖像修復（True/False）
        
        只返回存在的參數，不存在的參數請不要包含在返回結果中。
        請確保返回的是有效的 Python 字典格式。
        """
        
        try:
            response = self.model.generate_content(prompt)
            # 解析返回的字典
            # 這裡需要根據實際的返回格式進行調整
            return eval(response.text)
        except Exception as e:
            self.logger.error(f"解析指令時出錯：{str(e)}")
            return None
            
    def get_operation_description(self, operation: str) -> str:
        """獲取操作的描述"""
        return self.operation_descriptions.get(operation, operation) 
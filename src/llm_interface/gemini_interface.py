import google.generativeai as genai
from typing import Dict, Any, List
import json
import time
from google.api_core import retry
import logging

class GeminiInterface:
    def __init__(self, api_key: str):
        """初始化Gemini API介面"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 定義可用的圖像處理操作
        self.available_operations = {
            "gamma": "調整圖像的伽馬值",
            "contrast": "調整圖像的對比度",
            "brightness": "調整圖像的亮度",
            "saturation": "調整圖像的飽和度",
            "auto_wb": "自動白平衡",
            "denoise": "降噪處理",
            "sharpen": "銳化處理"
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

    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """解析使用者的自然語言指令"""
        try:
            # 簡單的關鍵字匹配
            operations = {}
            
            # 檢查白平衡相關指令
            if any(keyword in instruction.lower() for keyword in ["白平衡", "顏色", "色調", "色溫"]):
                operations["auto_wb"] = True
            
            # 檢查降噪相關指令
            if any(keyword in instruction.lower() for keyword in ["雜訊", "噪點", "降噪", "去噪"]):
                operations["denoise"] = True
            
            # 檢查銳化相關指令
            if any(keyword in instruction.lower() for keyword in ["清晰", "銳化", "增強"]):
                operations["sharpen"] = True
            
            # 檢查亮度和對比度相關指令
            if any(keyword in instruction.lower() for keyword in ["亮", "暗"]):
                operations["brightness"] = 10 if "亮" in instruction else -10
            if any(keyword in instruction.lower() for keyword in ["對比度", "對比"]):
                operations["contrast"] = 1.2
            
            # 如果沒有匹配到任何操作，使用 Gemini API
            if not operations:
                prompt = f"""
                你是一個圖像處理助手。請根據以下指令，判斷需要執行哪些圖像處理操作。
                可用的操作包括：
                {json.dumps(self.available_operations, ensure_ascii=False, indent=2)}

                請以JSON格式返回需要執行的操作及其參數。例如：
                {{
                    "operations": {{
                        "gamma": 1.2,
                        "contrast": 1.1,
                        "brightness": 10,
                        "saturation": 1.2,
                        "auto_wb": true,
                        "denoise": true,
                        "sharpen": true
                    }}
                }}

                使用者指令：{instruction}
                """
                
                response_text = self._call_gemini_api(prompt)
                result = json.loads(response_text)
                operations = result.get("operations", {})
            
            return operations
            
        except json.JSONDecodeError:
            self.logger.error("JSON 解析錯誤")
            return {}
        except Exception as e:
            self.logger.error(f"指令解析錯誤: {str(e)}")
            return {}

    def get_operation_description(self, operation: str) -> str:
        """獲取操作的描述"""
        return self.available_operations.get(operation, "未知操作") 
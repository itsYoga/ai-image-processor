"""
Style Transfer Models
風格轉換模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

class StyleTransfer:
    def __init__(self, model_type='cyclegan'):
        """
        初始化風格轉換模型
        
        Args:
            model_type: 模型類型 ('cyclegan' 或 'gan')
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'cyclegan':
            self.model = self._load_cyclegan()
        elif model_type == 'gan':
            self.model = self._load_gan()
        else:
            raise ValueError(f"不支援的模型類型: {model_type}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 定義圖像轉換
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_cyclegan(self):
        """載入CycleGAN模型"""
        # 這裡需要添加CycleGAN的模型架構和權重載入邏輯
        model = CycleGAN()
        return model
    
    def _load_gan(self):
        """載入GAN模型"""
        # 這裡需要添加GAN的模型架構和權重載入邏輯
        model = GAN()
        return model
    
    def transfer_style(self, image, style_type):
        """
        轉換圖像風格
        
        Args:
            image: 輸入圖像 (numpy array)
            style_type: 目標風格類型
            
        Returns:
            風格轉換後的圖像
        """
        with torch.no_grad():
            # 轉換為tensor
            img = self.transform(image).unsqueeze(0).to(self.device)
            
            # 執行風格轉換
            output = self.model(img, style_type)
            
            # 轉換回numpy array
            output = output.squeeze().cpu().numpy()
            output = output.transpose(1, 2, 0)
            output = (output * 255.0).round().astype('uint8')
            
            return output

class CycleGAN(nn.Module):
    """CycleGAN模型架構"""
    def __init__(self):
        super(CycleGAN, self).__init__()
        # 生成器
        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x, style_type):
        return self.generator(x)

class GAN(nn.Module):
    """GAN模型架構"""
    def __init__(self):
        super(GAN, self).__init__()
        # 生成器
        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, style_type):
        return self.generator(x) 
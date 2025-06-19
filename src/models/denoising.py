"""
Deep Learning Denoising Models
深度學習降噪模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Denoising:
    def __init__(self, model_type='dncnn'):
        """
        初始化降噪模型
        
        Args:
            model_type: 模型類型 ('dncnn' 或 'unet')
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'dncnn':
            self.model = self._load_dncnn()
        elif model_type == 'unet':
            self.model = self._load_unet()
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
    
    def _load_dncnn(self):
        """載入DnCNN模型"""
        model = DnCNN()
        # 這裡需要添加DnCNN的預訓練權重載入邏輯
        return model
    
    def _load_unet(self):
        """載入UNet模型"""
        model = UNet()
        # 這裡需要添加UNet的預訓練權重載入邏輯
        return model
    
    def denoise(self, image, noise_level=25):
        """
        降噪處理
        
        Args:
            image: 輸入圖像 (numpy array)
            noise_level: 噪聲水平
            
        Returns:
            降噪後的圖像
        """
        with torch.no_grad():
            # 轉換為tensor
            img = self.transform(image).unsqueeze(0).to(self.device)
            
            # 執行降噪
            output = self.model(img)
            
            # 轉換回numpy array
            output = output.squeeze().cpu().numpy()
            output = output.transpose(1, 2, 0)
            output = (output * 255.0).round().astype('uint8')
            
            return output

class DnCNN(nn.Module):
    """DnCNN模型架構"""
    def __init__(self, depth=17, n_channels=64, image_channels=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        
        # 第一層
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, 
                              kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 中間層
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, 
                                  kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # 最後一層
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, 
                              kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.dncnn(x)
        return out

class UNet(nn.Module):
    """UNet模型架構"""
    def __init__(self):
        super(UNet, self).__init__()
        
        # 編碼器
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # 解碼器
        self.dec4 = self._block(512, 256)
        self.dec3 = self._block(256, 128)
        self.dec2 = self._block(128, 64)
        self.dec1 = self._block(64, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 編碼
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 解碼
        d4 = self.dec4(self.up(e4))
        d3 = self.dec3(self.up(d4))
        d2 = self.dec2(self.up(d3))
        d1 = self.dec1(self.up(d2))
        
        return torch.sigmoid(d1) 
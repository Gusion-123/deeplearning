import torch
import torch.nn as nn

class AlzheimerNet3D(nn.Module):
    def __init__(self):
        super(AlzheimerNet3D, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个3D卷积块
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # 第二个3D卷积块
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # 第三个3D卷积块
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # 注意力机制
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # 3个类别
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 
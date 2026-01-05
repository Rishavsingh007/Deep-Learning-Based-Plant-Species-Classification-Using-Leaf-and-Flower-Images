"""
CT7160NI Computer Vision Coursework
Baseline CNN Architecture

This module implements a custom CNN architecture from scratch for
plant species classification. This serves as a baseline to compare
against transfer learning approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of:
    Conv2D -> BatchNorm -> ReLU -> MaxPool
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolutional kernel (default: 3)
    pool : bool
        Whether to apply max pooling (default: True)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class BaselineCNN(nn.Module):
    """
    Custom Baseline CNN Architecture for Plant Species Classification.
    
    Architecture:
    -------------
    Input (224x224x3)
        ↓
    ConvBlock(64) -> 112x112x64
        ↓
    ConvBlock(128) -> 56x56x128
        ↓
    ConvBlock(256) -> 28x28x256
        ↓
    ConvBlock(512) -> 14x14x512
        ↓
    Global Average Pooling -> 512
        ↓
    Dense(512) -> ReLU -> Dropout(0.5)
        ↓
    Dense(num_classes)
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes (default: 102 for Oxford Flowers)
    in_channels : int
        Number of input channels (default: 3 for RGB)
    dropout : float
        Dropout probability (default: 0.5)
    improved : bool
        If True, uses improved architecture with more capacity (default: False)
        
    Example:
    --------
    >>> model = BaselineCNN(num_classes=102)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([1, 102])
    """
    
    def __init__(self, num_classes=102, in_channels=3, dropout=0.5, improved=False):
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        self.improved = improved
        
        if improved:
            # Improved architecture with more capacity
            # Feature extraction layers - deeper with more channels
            self.features = nn.Sequential(
                ConvBlock(in_channels, 96, pool=False),   # 224 -> 224
                ConvBlock(96, 96),                         # 224 -> 112
                ConvBlock(96, 192, pool=False),            # 112 -> 112
                ConvBlock(192, 192),                       # 112 -> 56
                ConvBlock(192, 384, pool=False),           # 56 -> 56
                ConvBlock(384, 384),                       # 56 -> 28
                ConvBlock(384, 768, pool=False),          # 28 -> 28
                ConvBlock(768, 768),                       # 28 -> 14
            )
            
            # Global Average Pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Improved classification head with more capacity
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(768, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout * 0.5),  # Less dropout in second layer
                nn.Linear(512, num_classes)
            )
        else:
            # Original architecture
            # Feature extraction layers
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64),   # 224 -> 112
                ConvBlock(64, 128),           # 112 -> 56
                ConvBlock(128, 256),          # 56 -> 28
                ConvBlock(256, 512),          # 28 -> 14
            )
            
            # Global Average Pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes)
            )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """
        Extract features before the classification head.
        Useful for visualization and transfer learning.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_feature_maps(self, x):
        """
        Extract feature maps before global pooling.
        Useful for visualization.
        """
        return self.features(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineCNNDeep(nn.Module):
    """
    A deeper version of the baseline CNN with more layers.
    
    This version adds additional convolutional layers within each stage
    for improved feature extraction.
    """
    
    def __init__(self, num_classes=102, in_channels=3, dropout=0.5):
        super(BaselineCNNDeep, self).__init__()
        
        self.num_classes = num_classes
        
        # Stage 1: 224 -> 112
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 64, pool=False),
            ConvBlock(64, 64, pool=True),
        )
        
        # Stage 2: 112 -> 56
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, pool=False),
            ConvBlock(128, 128, pool=True),
        )
        
        # Stage 3: 56 -> 28
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=False),
            ConvBlock(256, 256, pool=True),
        )
        
        # Stage 4: 28 -> 14
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, pool=False),
            ConvBlock(512, 512, pool=False),
            ConvBlock(512, 512, pool=True),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing the model
if __name__ == "__main__":
    print("Testing BaselineCNN...")
    
    # Create model
    model = BaselineCNN(num_classes=102)
    
    # Print model summary
    print(f"\nModel Architecture:\n{model}")
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    # Test deeper model
    print("\n" + "="*50)
    print("Testing BaselineCNNDeep...")
    
    model_deep = BaselineCNNDeep(num_classes=102)
    print(f"Deep model parameters: {model_deep.count_parameters():,}")
    
    output_deep = model_deep(x)
    print(f"Deep model output shape: {output_deep.shape}")
    
    print("\nAll tests passed!")


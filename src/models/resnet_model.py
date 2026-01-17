"""
CT7160NI Computer Vision Coursework
ResNet Transfer Learning Model

This module implements transfer learning using ResNet50 pre-trained on ImageNet
for plant species classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetClassifier(nn.Module):
    """
    ResNet50 Transfer Learning Model for Plant Species Classification.
    
    This model uses a ResNet50 backbone pre-trained on ImageNet and
    replaces the final classification layer for our target task.
    
    Architecture:
    -------------
    ResNet50 Backbone (Pre-trained on ImageNet)
        ↓
    Adaptive Average Pooling -> 2048
        ↓
    Dense(512) -> ReLU -> Dropout
        ↓
    Dense(num_classes)
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes (default: 102 for Oxford Flowers)
    pretrained : bool
        Whether to use ImageNet pre-trained weights (default: True)
    freeze_backbone : bool
        Whether to freeze backbone layers (default: True for initial training)
    dropout : float
        Dropout probability (default: 0.3)
        
    Example:
    --------
    >>> model = ResNetClassifier(num_classes=102, pretrained=True)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([1, 102])
    """
    
    def __init__(self, num_classes=102, pretrained=True, freeze_backbone=True, dropout=0.3):
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            print("Loaded ResNet50 with ImageNet pre-trained weights")
        else:
            self.backbone = models.resnet50(weights=None)
            print("Loaded ResNet50 without pre-trained weights")
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone layers frozen")
        
    def unfreeze_backbone(self, unfreeze_from_layer=0):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Parameters:
        -----------
        unfreeze_from_layer : int
            Layer index to start unfreezing from (0 = all layers)
        """
        # Get all named parameters
        layers = list(self.backbone.named_parameters())
        
        for i, (name, param) in enumerate(layers):
            if i >= unfreeze_from_layer:
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"Unfroze backbone from layer {unfreeze_from_layer}")
        print(f"Trainable backbone params: {trainable:,} / {total:,}")
        
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features before the classification head."""
        return self.backbone(x)
    
    def count_parameters(self, trainable_only=True):
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ResNetClassifierWithAttention(nn.Module):
    """
    ResNet50 with Channel Attention Module (SE Block).
    
    Adds Squeeze-and-Excitation attention to improve feature weighting.
    """
    
    def __init__(self, num_classes=102, pretrained=True, dropout=0.3):
        super(ResNetClassifierWithAttention, self).__init__()
        
        # Load pre-trained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Squeeze-and-Excitation attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Get features from backbone (before global avg pool)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply attention
        batch_size = x.size(0)
        attention_weights = self.attention(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = x.view(batch_size, -1)
        
        # Apply attention weights
        x = x * attention_weights
        
        # Classification
        output = self.classifier(x)
        return output


# Testing the model
if __name__ == "__main__":
    print("Testing ResNetClassifier...")
    
    # Create model with frozen backbone
    model = ResNetClassifier(num_classes=102, pretrained=True, freeze_backbone=True)
    
    print(f"\nTrainable parameters (frozen): {model.count_parameters(trainable_only=True):,}")
    print(f"Total parameters: {model.count_parameters(trainable_only=False):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test unfreezing
    print("\n" + "="*50)
    print("Unfreezing backbone...")
    model.unfreeze_backbone(unfreeze_from_layer=0)
    print(f"Trainable parameters (unfrozen): {model.count_parameters(trainable_only=True):,}")
    
    # Test attention model
    print("\n" + "="*50)
    print("Testing ResNetClassifierWithAttention...")
    
    model_attention = ResNetClassifierWithAttention(num_classes=102)
    output_attention = model_attention(x)
    print(f"Attention model output shape: {output_attention.shape}")
    
    print("\nAll tests passed!")


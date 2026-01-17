"""
CT7160NI Computer Vision Coursework
EfficientNet Transfer Learning Model

This module implements transfer learning using EfficientNet pre-trained on ImageNet
for plant species classification. EfficientNet achieves state-of-the-art accuracy
with fewer parameters through compound scaling.
"""

import torch
import torch.nn as nn
import timm  # PyTorch Image Models library


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet Transfer Learning Model for Plant Species Classification.
    
    EfficientNet uses compound scaling to efficiently scale network width,
    depth, and resolution for better accuracy with fewer parameters.
    
    Variants:
    ---------
    - EfficientNet-B0: 5.3M params, 224x224 input
    - EfficientNet-B3: 12M params, 300x300 input (recommended)
    - EfficientNet-B7: 66M params, 600x600 input
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes (default: 102)
    model_name : str
        EfficientNet variant (default: 'efficientnet_b3')
    pretrained : bool
        Whether to use ImageNet pre-trained weights (default: True)
    freeze_backbone : bool
        Whether to freeze backbone layers (default: True)
    dropout : float
        Dropout probability (default: 0.3)
        
    Example:
    --------
    >>> model = EfficientNetClassifier(num_classes=102)
    >>> x = torch.randn(1, 3, 300, 300)  # Use 300x300 for B3
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([1, 102])
    """
    
    def __init__(
        self, 
        num_classes=102, 
        model_name='efficientnet_b3',
        pretrained=True, 
        freeze_backbone=True, 
        dropout=0.3
    ):
        super(EfficientNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained EfficientNet using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Keep global average pooling
        )
        
        if pretrained:
            print(f"Loaded {model_name} with ImageNet pre-trained weights")
        
        # Get number of features from backbone
        num_features = self.backbone.num_features
        print(f"Backbone output features: {num_features}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
        
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone layers frozen")
        
    def unfreeze_backbone(self, unfreeze_ratio=1.0):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Parameters:
        -----------
        unfreeze_ratio : float
            Ratio of layers to unfreeze (1.0 = all, 0.5 = last half)
        """
        params = list(self.backbone.parameters())
        num_params = len(params)
        unfreeze_from = int(num_params * (1 - unfreeze_ratio))
        
        for i, param in enumerate(params):
            if i >= unfreeze_from:
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Unfroze {unfreeze_ratio*100:.0f}% of backbone")
        print(f"Trainable backbone params: {trainable:,}")
        
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
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
    
    @staticmethod
    def get_recommended_input_size(model_name):
        """Get recommended input size for each EfficientNet variant."""
        sizes = {
            'efficientnet_b0': 224,
            'efficientnet_b1': 240,
            'efficientnet_b2': 260,
            'efficientnet_b3': 300,
            'efficientnet_b4': 380,
            'efficientnet_b5': 456,
            'efficientnet_b6': 528,
            'efficientnet_b7': 600,
        }
        return sizes.get(model_name, 224)


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2 Transfer Learning Model.
    
    EfficientNetV2 is an improved version with faster training and
    better parameter efficiency.
    """
    
    def __init__(
        self, 
        num_classes=102, 
        model_name='efficientnetv2_s',  # s, m, or l
        pretrained=True,
        dropout=0.3
    ):
        super(EfficientNetV2Classifier, self).__init__()
        
        # Load EfficientNetV2
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def create_model(model_type, num_classes=102, pretrained=True, **kwargs):
    """
    Factory function to create different model architectures.
    
    Parameters:
    -----------
    model_type : str
        One of 'efficientnet_b0', 'efficientnet_b3', 'efficientnetv2_s', etc.
    num_classes : int
        Number of output classes
    pretrained : bool
        Whether to use pre-trained weights
        
    Returns:
    --------
    nn.Module : The created model
    """
    if model_type.startswith('efficientnetv2'):
        return EfficientNetV2Classifier(
            num_classes=num_classes,
            model_name=model_type,
            pretrained=pretrained,
            **kwargs
        )
    else:
        return EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_type,
            pretrained=pretrained,
            **kwargs
        )


# Testing the model
if __name__ == "__main__":
    print("Testing EfficientNetClassifier...")
    
    # Test EfficientNet-B3
    model = EfficientNetClassifier(
        num_classes=102, 
        model_name='efficientnet_b3',
        pretrained=True,
        freeze_backbone=True
    )
    
    # Get recommended input size
    input_size = EfficientNetClassifier.get_recommended_input_size('efficientnet_b3')
    print(f"Recommended input size: {input_size}x{input_size}")
    
    print(f"\nTrainable parameters (frozen): {model.count_parameters(trainable_only=True):,}")
    print(f"Total parameters: {model.count_parameters(trainable_only=False):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, input_size, input_size)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test unfreezing
    print("\n" + "="*50)
    print("Unfreezing 50% of backbone...")
    model.unfreeze_backbone(unfreeze_ratio=0.5)
    print(f"Trainable parameters (partially unfrozen): {model.count_parameters(trainable_only=True):,}")
    
    # Test different variants
    print("\n" + "="*50)
    print("Available EfficientNet variants:")
    for variant in ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b7']:
        size = EfficientNetClassifier.get_recommended_input_size(variant)
        print(f"  {variant}: {size}x{size}")
    
    print("\nAll tests passed!")


"""
Plant Species Classification - Architecture Diagrams
Generates exportable diagrams for the project proposal
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def draw_system_architecture():
    """Draw the overall system pipeline architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Plant Species Classification - System Architecture', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'preprocess': '#FFF3E0', 
        'model': '#E8F5E9',
        'output': '#FCE4EC',
        'arrow': '#37474F'
    }
    
    # Box style
    box_style = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#333333', linewidth=2)
    
    # === INPUT SECTION ===
    input_box = FancyBboxPatch((0.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], edgecolor='#1565C0', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 6.8, '📷 Input', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 6.3, 'Raw Flower Image', fontsize=10, ha='center')
    ax.text(1.75, 5.9, '(Oxford 102 Dataset)', fontsize=9, ha='center', style='italic')
    
    # === PREPROCESSING SECTION ===
    preprocess_box = FancyBboxPatch((4, 4.5), 3, 3.5, boxstyle="round,pad=0.1",
                                     facecolor=colors['preprocess'], edgecolor='#EF6C00', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(5.5, 7.5, '⚙️ Preprocessing', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.5, 6.9, '• Resize (224×224)', fontsize=9, ha='center')
    ax.text(5.5, 6.5, '• Normalization', fontsize=9, ha='center')
    ax.text(5.5, 6.1, '• Data Augmentation:', fontsize=9, ha='center')
    ax.text(5.5, 5.7, '  - Random Rotation', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 5.35, '  - Horizontal Flip', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 5.0, '  - Color Jitter', fontsize=8, ha='center', style='italic')
    
    # === MODEL SECTION ===
    model_box = FancyBboxPatch((8, 3), 3.5, 5, boxstyle="round,pad=0.1",
                                facecolor=colors['model'], edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(model_box)
    ax.text(9.75, 7.5, '🧠 CNN Models', fontsize=12, fontweight='bold', ha='center')
    
    # Sub-boxes for models
    for i, (model, acc) in enumerate([('Baseline CNN', '~68%'), 
                                       ('ResNet50', '~85%'), 
                                       ('EfficientNet-B3', '~89%')]):
        y_pos = 6.5 - i * 1.2
        sub_box = FancyBboxPatch((8.3, y_pos - 0.4), 2.9, 0.9, boxstyle="round,pad=0.05",
                                  facecolor='white', edgecolor='#4CAF50', linewidth=1.5)
        ax.add_patch(sub_box)
        ax.text(9.75, y_pos + 0.1, model, fontsize=10, ha='center', fontweight='bold')
        ax.text(9.75, y_pos - 0.2, f'Accuracy: {acc}', fontsize=8, ha='center', color='#666')
    
    # === OUTPUT SECTION ===
    output_box = FancyBboxPatch((12, 5), 1.8, 2.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='#C2185B', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.9, 7.0, '📊 Output', fontsize=12, fontweight='bold', ha='center')
    ax.text(12.9, 6.4, 'Species', fontsize=10, ha='center')
    ax.text(12.9, 6.0, 'Prediction', fontsize=10, ha='center')
    ax.text(12.9, 5.4, '(1 of 102)', fontsize=9, ha='center', style='italic')
    
    # === ARROWS ===
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2.5, 
                       connectionstyle='arc3,rad=0')
    
    ax.annotate('', xy=(4, 6.5), xytext=(3, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(8, 6.5), xytext=(7, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(12, 6.25), xytext=(11.5, 6.25), arrowprops=arrow_style)
    
    # === EVALUATION METRICS BOX ===
    eval_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#F5F5F5', edgecolor='#616161', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(7, 2.6, '📈 Evaluation Metrics', fontsize=12, fontweight='bold', ha='center')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confusion Matrix', 'Grad-CAM']
    for i, metric in enumerate(metrics):
        x_pos = 1.5 + i * 2.1
        metric_box = FancyBboxPatch((x_pos - 0.8, 0.8), 1.8, 1.2, boxstyle="round,pad=0.05",
                                     facecolor='white', edgecolor='#9E9E9E', linewidth=1)
        ax.add_patch(metric_box)
        ax.text(x_pos + 0.1, 1.4, metric, fontsize=9, ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def draw_cnn_architecture():
    """Draw the detailed CNN model architectures"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    fig.suptitle('Model Architectures Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for layers
    layer_colors = {
        'conv': '#BBDEFB',
        'pool': '#C8E6C9',
        'fc': '#FFE0B2',
        'dropout': '#F8BBD9',
        'output': '#E1BEE7',
        'backbone': '#B2DFDB'
    }
    
    def draw_layer(ax, y, width, height, color, label, sublabel=''):
        box = FancyBboxPatch((0.5 - width/2, y), width, height, 
                              boxstyle="round,pad=0.02", facecolor=color, 
                              edgecolor='#333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(0.5, y + height/2 + 0.1, label, fontsize=9, ha='center', 
                fontweight='bold', va='center')
        if sublabel:
            ax.text(0.5, y + height/2 - 0.15, sublabel, fontsize=7, ha='center', 
                    va='center', style='italic')
    
    def draw_arrow(ax, y_start, y_end):
        ax.annotate('', xy=(0.5, y_end + 0.02), xytext=(0.5, y_start - 0.02),
                   arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.5))
    
    # === BASELINE CNN ===
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Baseline CNN\n(Custom Architecture)', fontsize=12, fontweight='bold', pad=10)
    
    layers_baseline = [
        (9.0, 0.6, layer_colors['conv'], 'Input', '224×224×3'),
        (8.2, 0.6, layer_colors['conv'], 'Conv2D + BN + ReLU', '64 filters, 3×3'),
        (7.4, 0.4, layer_colors['pool'], 'MaxPool 2×2', ''),
        (6.6, 0.6, layer_colors['conv'], 'Conv2D + BN + ReLU', '128 filters, 3×3'),
        (5.8, 0.4, layer_colors['pool'], 'MaxPool 2×2', ''),
        (5.0, 0.6, layer_colors['conv'], 'Conv2D + BN + ReLU', '256 filters, 3×3'),
        (4.2, 0.4, layer_colors['pool'], 'MaxPool 2×2', ''),
        (3.4, 0.6, layer_colors['conv'], 'Conv2D + BN + ReLU', '512 filters, 3×3'),
        (2.6, 0.4, layer_colors['pool'], 'Global Avg Pool', ''),
        (1.8, 0.5, layer_colors['fc'], 'Dense + ReLU', '512 units'),
        (1.1, 0.4, layer_colors['dropout'], 'Dropout', '0.5'),
        (0.4, 0.5, layer_colors['output'], 'Dense + Softmax', '102 classes'),
    ]
    
    for i, (y, h, color, label, sublabel) in enumerate(layers_baseline):
        draw_layer(ax1, y, 0.85, h, color, label, sublabel)
        if i < len(layers_baseline) - 1:
            draw_arrow(ax1, y, layers_baseline[i+1][0] + layers_baseline[i+1][1])
    
    ax1.text(0.5, 0.05, '~2.5M Parameters', fontsize=10, ha='center', 
             fontweight='bold', color='#1565C0')
    
    # === RESNET50 ===
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('ResNet50\n(Transfer Learning)', fontsize=12, fontweight='bold', pad=10)
    
    layers_resnet = [
        (9.0, 0.6, layer_colors['conv'], 'Input', '224×224×3'),
        (6.5, 2.2, layer_colors['backbone'], 'ResNet50 Backbone', 'Pre-trained (ImageNet)\nLayers 1-45: FROZEN'),
        (5.7, 0.5, layer_colors['pool'], 'AdaptiveAvgPool', '(1,1)'),
        (4.9, 0.5, layer_colors['fc'], 'Dense + ReLU', '512 units'),
        (4.2, 0.4, layer_colors['dropout'], 'Dropout', '0.3'),
        (3.5, 0.5, layer_colors['output'], 'Dense + Softmax', '102 classes'),
    ]
    
    for i, (y, h, color, label, sublabel) in enumerate(layers_resnet):
        draw_layer(ax2, y, 0.85, h, color, label, sublabel)
        if i < len(layers_resnet) - 1:
            draw_arrow(ax2, y, layers_resnet[i+1][0] + layers_resnet[i+1][1])
    
    # Skip connection illustration
    ax2.annotate('', xy=(0.15, 7.0), xytext=(0.15, 8.2),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    ax2.text(0.08, 7.6, 'Skip\nConn.', fontsize=7, ha='center', color='#E65100')
    
    ax2.text(0.5, 0.05, '~23.5M Params (2M trainable)', fontsize=10, ha='center',
             fontweight='bold', color='#1565C0')
    
    # === EFFICIENTNET-B3 ===
    ax3 = axes[2]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('EfficientNet-B3\n(Transfer Learning)', fontsize=12, fontweight='bold', pad=10)
    
    layers_efficient = [
        (9.0, 0.6, layer_colors['conv'], 'Input', '300×300×3'),
        (6.2, 2.5, layer_colors['backbone'], 'EfficientNet-B3', 'Pre-trained (ImageNet)\nMBConv + SE Blocks\nCompound Scaling'),
        (5.4, 0.5, layer_colors['pool'], 'Global Avg Pool', ''),
        (4.6, 0.5, layer_colors['fc'], 'Dense + ReLU', '256 units'),
        (3.9, 0.4, layer_colors['dropout'], 'Dropout', '0.3'),
        (3.2, 0.5, layer_colors['output'], 'Dense + Softmax', '102 classes'),
    ]
    
    for i, (y, h, color, label, sublabel) in enumerate(layers_efficient):
        draw_layer(ax3, y, 0.85, h, color, label, sublabel)
        if i < len(layers_efficient) - 1:
            draw_arrow(ax3, y, layers_efficient[i+1][0] + layers_efficient[i+1][1])
    
    ax3.text(0.5, 0.05, '~10.7M Parameters', fontsize=10, ha='center',
             fontweight='bold', color='#1565C0')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=layer_colors['conv'], edgecolor='#333', label='Convolution'),
        mpatches.Patch(facecolor=layer_colors['pool'], edgecolor='#333', label='Pooling'),
        mpatches.Patch(facecolor=layer_colors['backbone'], edgecolor='#333', label='Pre-trained Backbone'),
        mpatches.Patch(facecolor=layer_colors['fc'], edgecolor='#333', label='Fully Connected'),
        mpatches.Patch(facecolor=layer_colors['dropout'], edgecolor='#333', label='Dropout'),
        mpatches.Patch(facecolor=layer_colors['output'], edgecolor='#333', label='Output'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


def draw_data_pipeline():
    """Draw the data preprocessing pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Data Preprocessing & Augmentation Pipeline', 
                 fontsize=14, fontweight='bold', pad=15)
    
    colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9', '#FCE4EC', '#F3E5F5']
    
    stages = [
        ('Raw Images\n(8,189 images)', '📁'),
        ('Load &\nDecode', '📥'),
        ('Resize\n(224×224)', '📐'),
        ('Augmentation\n(Training Only)', '🔄'),
        ('Normalize\n(ImageNet Stats)', '📊'),
        ('Tensor\nConversion', '🔢'),
        ('DataLoader\n(Batches)', '📦')
    ]
    
    for i, (label, icon) in enumerate(stages):
        x = 1 + i * 1.8
        color = colors[i % len(colors)]
        box = FancyBboxPatch((x - 0.6, 2), 1.4, 2.5, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.1, 4.0, icon, fontsize=20, ha='center', va='center')
        ax.text(x + 0.1, 2.8, label, fontsize=9, ha='center', va='center', fontweight='bold')
        
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 0.95, 3.25), xytext=(x + 0.65, 3.25),
                       arrowprops=dict(arrowstyle='->', color='#37474F', lw=2))
    
    # Add augmentation details
    aug_box = FancyBboxPatch((5.5, 0.3), 5.5, 1.4, boxstyle="round,pad=0.1",
                              facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=1.5)
    ax.add_patch(aug_box)
    ax.text(8.25, 1.35, 'Augmentation Techniques:', fontsize=10, ha='center', fontweight='bold')
    augs = '• Random Rotation (±20°)  • Horizontal Flip  • Vertical Flip  • Color Jitter  • Random Crop'
    ax.text(8.25, 0.7, augs, fontsize=8, ha='center')
    
    # Connect augmentation box
    ax.plot([6.5, 6.5], [1.7, 2.0], color='#FF8F00', lw=1.5, linestyle='--')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Generating architecture diagrams...")
    
    # Generate and save all diagrams
    fig1 = draw_system_architecture()
    fig1.savefig('system_architecture.png', dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig1.savefig('system_architecture.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: system_architecture.png/pdf")
    
    fig2 = draw_cnn_architecture()
    fig2.savefig('model_architectures.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig('model_architectures.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: model_architectures.png/pdf")
    
    fig3 = draw_data_pipeline()
    fig3.savefig('data_pipeline.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig3.savefig('data_pipeline.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✓ Saved: data_pipeline.png/pdf")
    
    print("\n✅ All diagrams generated successfully!")
    print("Files created:")
    print("  - system_architecture.png/pdf")
    print("  - model_architectures.png/pdf") 
    print("  - data_pipeline.png/pdf")
    
    plt.show()

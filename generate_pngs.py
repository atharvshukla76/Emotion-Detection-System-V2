import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_snake(nodes, filename, box_color, edge_color, num_cols=3):
    n = len(nodes)
    num_rows = (n + num_cols - 1) // num_cols
    
    # Increased box dimensions and spacing to prevent text overflow
    box_w = 5.2
    box_h = 3.0
    x_spacing = 6.6
    y_spacing = 4.4
    
    actual_cols = min(n, num_cols)
    fig_w = actual_cols * x_spacing
    fig_h = num_rows * y_spacing
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-fig_h + 1.4, 3.4)
    ax.axis('off')
    
    positions = []
    for i in range(n):
        row = i // num_cols
        col = i % num_cols
        if row % 2 == 1:
            col = (num_cols - 1) - col # reverse
            
        x = col * x_spacing + 0.7
        y = -row * y_spacing
        positions.append((x, y))
        
    for i, (x, y) in enumerate(positions):
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.1,rounding_size=0.1", 
                                      linewidth=4, edgecolor=edge_color, facecolor=box_color)
        ax.add_patch(rect)
        
        ax.text(x + box_w/2, y + box_h/2, nodes[i], ha='center', va='center', 
                fontsize=20, fontfamily='sans-serif', fontweight='bold', color='black')
        
        if i < n - 1:
            nx, ny = positions[i+1]
            row = i // num_cols
            
            # if moving down (end of a row)
            if i % num_cols == num_cols - 1:
                ax.annotate('', xy=(x + box_w/2, ny + box_h), xytext=(x + box_w/2, y),
                            arrowprops=dict(arrowstyle="->,head_width=0.8,head_length=1.0", lw=4, color='black'))
            else:
                # moving horizontally
                if row % 2 == 0:
                    # moving right
                    ax.annotate('', xy=(nx, y + box_h/2), xytext=(x + box_w, y + box_h/2),
                                arrowprops=dict(arrowstyle="->,head_width=0.8,head_length=1.0", lw=4, color='black'))
                else:
                    # moving left
                    ax.annotate('', xy=(nx + box_w, y + box_h/2), xytext=(x, y + box_h/2),
                                arrowprops=dict(arrowstyle="->,head_width=0.8,head_length=1.0", lw=4, color='black'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

nodes_5 = [
    "Video Input\n15 frames",
    "Grayscale & Optical Flow\nResize 64x64, stack 30\nchannels",
    "Stage 1\nConv2D (32, 3x3) ->\nBatchNorm\n-> ReLU -> MaxPool 2x2",
    "Stage 2\nConv2D(64, 3x3) ->\nBatchNorm\n-> ReLU -> MaxPool 2x2",
    "Stage 3\nConv2D (64, 3x3) ->\nBatchNorm\n-> ReLU -> Dropout",
    "Global Average Pooling\n2D",
    "Dense Layer 64\nNeurons",
    "Video Feature Vector\nV ∈ R^64"
]

nodes_6 = [
    "Audio Input",
    "Log-Mel Spectrogram",
    "Whisper Encoder\nTransformer Blocks",
    "Whisper Decoder\nTransformer Blocks",
    "Transcribed Text",
    "Byte-Pair Encoding\nBPE Tokenization",
    "Token + Position\nEmbeddings",
    "Transformer Layers x6\n-- Multi-Head\nSelf-Attention\n--LayerNorm &\nFeed Forward",
    "CLS Token\nExtraction",
    "Text Feature Vector\nT ∈ R^768"
]

nodes_7 = [
    "Audio Input 3 sec Clip\n22050Hz",
    "Librosa Preprocessing\nMel-Spectrogram\n+ MFCC\n150 x 156 x 1 Matrix",
    "Stage 1\nConv2D (32, 3x3) ->\nMaxPool\nConv2D (64, 3x3) ->\nMaxPool",
    "Stage 2\nConv2D (32, 3x3) ->\nMaxPool\nConv2D (64, 3x3) ->\nMaxPool",
    "Stage 3\nConv2D (64, 3x3) ->\nMaxPool\nConv2D (64, 3x3)",
    "Stage 4\nConv2D (64, 3x3) ->\nMaxPool\nConv2D (64, 3x3)",
    "Reshape to\n1D Sequence",
    "Stage 5\nConv1D (64, 1x1)\nConv1D (32, 3x3)",
    "Global Average\nPooling 1D",
    "Dense Layer 64\nNeurons",
    "Audio Feature Vector\nA ∈ R^64"
]

nodes_8 = [
    "Cropped Face\nSingle Frame",
    "Initial Convolution\nConv 7x7 ->\nBatchNorm -> ReLU\n-> MaxPool",
    "Stage 1:\nResidual Block x3\nBottleneck: Conv 1x1 ->\nConv 3x3 -> Conv 1x1 +\nResidual Skip\nConnection",
    "Stage 2:\nResidual Block x4\nBottleneck: Conv 1x1 ->\nConv 3x3 -> Conv 1x1 +\nResidual Skip\nConnection",
    "Stage 3:\nResidual Block x6\nBottleneck: Conv 1x1 ->\nConv 3x3 -> Conv 1x1 +\nResidual Skip\nConnection",
    "Stage 4:\nResidual Block x3\nBottleneck: Conv 1x1 ->\nConv 3x3 -> Conv 1x1 +\nResidual Skip\nConnection",
    "Global Average\nPooling",
    "Fully Connected\nLayer",
    "Static Feature Vector\nS ∈ R^2048"
]

draw_snake(nodes_5, 'Diagram_5_Video_CNN.png', '#bbdefb', '#1565c0')
draw_snake(nodes_6, 'Diagram_6_NLP.png', '#e1bee7', '#6a1b9a')
draw_snake(nodes_7, 'Diagram_7_Audio_CNN.png', '#c8e6c9', '#2e7d32')
draw_snake(nodes_8, 'Diagram_8_ResNet50.png', '#ffe0b2', '#ef6c00')

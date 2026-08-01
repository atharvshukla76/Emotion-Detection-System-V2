import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_l_shape(nodes, filename, box_color, edge_color):
    n = len(nodes)
    h_nodes = (n + 1) // 2
    v_nodes = n - h_nodes
    
    fig, ax = plt.subplots(figsize=(h_nodes * 4.0, v_nodes * 2.5 + 2))
    ax.set_xlim(0, h_nodes * 4.0)
    ax.set_ylim(-v_nodes * 2.5 - 0.5, 2.5)
    ax.axis('off')
    
    box_w = 3.4
    box_h = 2.0
    
    positions = []
    # Horizontal arm
    for i in range(h_nodes):
        x = i * 4.0 + 0.3
        y = 0
        positions.append((x, y))
    
    # Vertical arm (going down from the last horizontal node)
    last_x = positions[-1][0]
    for i in range(1, v_nodes + 1):
        x = last_x
        y = -i * 2.5
        positions.append((x, y))
        
    for i, (x, y) in enumerate(positions):
        # Draw box
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.1", 
                                      linewidth=2, edgecolor=edge_color, facecolor=box_color)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + box_w/2, y + box_h/2, nodes[i], ha='center', va='center', 
                fontsize=11, fontfamily='sans-serif', fontweight='bold', color='black')
        
        # Draw arrow to next
        if i < len(positions) - 1:
            nx, ny = positions[i+1]
            if i < h_nodes - 1: # Horizontal arrow
                ax.annotate('', xy=(nx, y + box_h/2), xytext=(x + box_w, y + box_h/2),
                            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.5, color='black'))
            else: # Vertical arrow
                ax.annotate('', xy=(x + box_w/2, ny + box_h), xytext=(x + box_w/2, y),
                            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.5, color='black'))

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
draw_l_shape(nodes_5, 'test_mpl.png', '#bbdefb', '#1565c0')

"""Generate architecture/pipeline diagrams for the 4 AzSL recognition models.

Produces four PNG files in this directory:
  - cosine_pipeline.png
  - dtw_pipeline.png
  - lstm_seq2seq_pipeline.png
  - transformer_pipeline.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def setup_axes(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, text, fc='#E8F0FE', ec='#1A73E8', fontsize=10,
        text_color='#0B2545', bold=False, rounded=0.02):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.02,rounding_size={rounded*100}",
                           linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight=weight, wrap=True)

def arrow(ax, x1, y1, x2, y2, color='#5F6368', lw=1.6, style='-|>'):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw)
    ax.add_patch(a)

def label(ax, x, y, text, fontsize=10, color='#202124', weight='normal',
          ha='center', va='center'):
    ax.text(x, y, text, fontsize=fontsize, color=color, weight=weight,
            ha=ha, va=va)

def title(ax, text):
    ax.text(50, 96, text, fontsize=15, weight='bold',
            color='#0B2545', ha='center', va='center')

# Color palette per stage type
C_INPUT   = ('#FFF4E5', '#F29900')  # video / camera input
C_PREPROC = ('#E6F4EA', '#137333')  # preprocessing
C_FEATURE = ('#E8F0FE', '#1A73E8')  # feature extraction
C_MODEL   = ('#FCE8E6', '#C5221F')  # model / core compute
C_OUTPUT  = ('#F3E8FD', '#8430CE')  # output / results
C_NOTE    = ('#F1F3F4', '#5F6368')  # note / params

# =============================================================================
# 1. COSINE PIPELINE
# =============================================================================
def make_cosine():
    fig, ax = setup_axes((13, 6.5))
    title(ax, 'Approach 1 — Cosine Similarity Retrieval Pipeline')

    # Top row — query path
    box(ax, 2,  72, 14, 12, 'Query video\n(.mp4)',          *C_INPUT, bold=True)
    box(ax, 20, 72, 16, 12, 'Uniform sampling\nN = 64 frames',*C_PREPROC)
    box(ax, 40, 72, 18, 12, 'MediaPipe Hands\n21 landmarks × (x,y,z) × 2', *C_FEATURE)
    box(ax, 62, 72, 18, 12, 'Wrist-centered\n+ scale norm. → 126-D / frame', *C_FEATURE)
    box(ax, 84, 72, 14, 12, 'Flatten\n64 × 126\n= 8064-D', *C_FEATURE, fontsize=9)

    arrow(ax, 16, 78, 20, 78)
    arrow(ax, 36, 78, 40, 78)
    arrow(ax, 58, 78, 62, 78)
    arrow(ax, 80, 78, 84, 78)

    # Down arrow into similarity engine
    arrow(ax, 91, 72, 91, 56)

    # Reference DB (bottom left)
    box(ax, 2, 38, 28, 18,
        'Reference database\n• 335 videos × 8064-D embeddings\n• Pre-computed offline\n• .npy cache',
        *C_PREPROC, fontsize=9)
    arrow(ax, 30, 47, 70, 47)

    # Similarity engine
    box(ax, 70, 38, 26, 18,
        'Cosine Similarity Engine\nsim(a,b) = (a·b) / (‖a‖‖b‖)\nper-sentence: max over refs',
        *C_MODEL, bold=True, fontsize=9)

    # Output
    arrow(ax, 83, 38, 83, 22)
    box(ax, 65, 8, 32, 14,
        'Top-K candidates\n(sentence, similarity score)',
        *C_OUTPUT, bold=True)

    # Footnote
    label(ax, 50, 2,
          'Training-free  ·  CPU-only  ·  search latency ~2.8 ms / query',
          fontsize=9, color='#5F6368')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'cosine_pipeline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'wrote {out}')

# =============================================================================
# 2. DTW PIPELINE (Method 4B)
# =============================================================================
def make_dtw():
    fig, ax = setup_axes((13, 7))
    title(ax, 'Approach 2 — DTW Pipeline (Method 4B: Pairwise Bone Angles)')

    # Row 1 — preprocessing
    box(ax, 2,  78, 14, 12, 'Query video',          *C_INPUT, bold=True)
    box(ax, 18, 78, 16, 12, 'MediaPipe Hands\nper-frame landmarks', *C_FEATURE)
    box(ax, 36, 78, 18, 12, 'Drop hand-less\n+ near-duplicate (>0.99)\nframes',     *C_PREPROC, fontsize=9)
    box(ax, 56, 78, 20, 12, 'Geometric features\n~190 pairwise bone angles\n× 2 hands → 380-D', *C_FEATURE, fontsize=9)
    box(ax, 78, 78, 18, 12, 'Hand-swap\nvariant\n(L↔R)', *C_PREPROC, fontsize=9)

    arrow(ax, 16, 84, 18, 84)
    arrow(ax, 34, 84, 36, 84)
    arrow(ax, 54, 84, 56, 84)
    arrow(ax, 76, 84, 78, 84)

    # Down arrow
    arrow(ax, 87, 78, 87, 64)

    # Subsequence DTW core
    box(ax, 30, 46, 60, 18,
        'Subsequence DTW alignment\n'
        'D[n,m] = C[n,m] + min(D[n-1,m-1], D[n-1,m], D[n,m-1])\n'
        'Sakoe–Chiba band ratio = 0.25',
        *C_MODEL, bold=True, fontsize=10)

    # Reference templates (left)
    box(ax, 2, 46, 24, 18,
        'Reference templates\n94 phrase exemplars\nsame 380-D feature\npre-computed',
        *C_PREPROC, fontsize=9)
    arrow(ax, 26, 55, 30, 55)

    # min cost selection
    arrow(ax, 60, 46, 60, 32)
    box(ax, 32, 18, 56, 14,
        'Score = min(normal, hand-swapped) cost\nrank templates by alignment cost',
        *C_MODEL, fontsize=10)

    # Output
    arrow(ax, 60, 18, 60, 8)
    box(ax, 32, -2, 56, 10,
        'Top-K phrases  +  alignment path (where the learner deviated)',
        *C_OUTPUT, bold=True, fontsize=10)

    label(ax, 50, -8,
          'Best overall: 40.0% Top-1 · 50.2% Top-3 · streaming variant for live camera',
          fontsize=9, color='#5F6368')

    ax.set_ylim(-12, 100)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'dtw_pipeline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'wrote {out}')

# =============================================================================
# 3. LSTM SEQ2SEQ PIPELINE
# =============================================================================
def make_lstm():
    fig, ax = setup_axes((13, 8))
    title(ax, 'Approach 3 — LSTM Seq2Seq with Bahdanau Attention')

    # Stage A title
    label(ax, 50, 90, 'Stage A — Offline Feature Caching', fontsize=11,
          weight='bold', color='#137333')

    box(ax, 2,  76, 12, 10, 'Video\nframes',           *C_INPUT)
    box(ax, 16, 76, 16, 10, 'MediaPipe\nhand-centric\n600×600 crop', *C_PREPROC, fontsize=9)
    box(ax, 34, 76, 14, 10, 'Resize\n224×224',         *C_PREPROC)
    box(ax, 50, 76, 18, 10, 'SqueezeNet 1.1\n(ImageNet)\n86 528-D / frame', *C_FEATURE, fontsize=9)
    box(ax, 70, 76, 14, 10, '1-layer\nBiLSTM\n(h = 256)', *C_FEATURE, fontsize=9)
    box(ax, 86, 76, 12, 10, 'Cache\n(T, 512)\n.pt file', *C_PREPROC, fontsize=9)

    for x in (14, 32, 48, 68, 84):
        arrow(ax, x, 81, x + 2, 81)

    # Stage B title
    label(ax, 50, 64, 'Stage B — Lightweight Seq2Seq Training on Cached Features',
          fontsize=11, weight='bold', color='#C5221F')

    # Encoder
    box(ax, 4, 44, 22, 16,
        'BiLSTM Encoder\nh = 512, bidirectional\nout: H = (T, 1024)',
        *C_FEATURE, bold=True, fontsize=10)

    # Attention
    box(ax, 32, 44, 22, 16,
        'Bahdanau attention\nα_{s,t} = softmax(eₛ,t / τ)\nτ = 2.0, dropout 0.1',
        *C_MODEL, bold=True, fontsize=10)
    arrow(ax, 26, 52, 32, 52)

    # Decoder
    box(ax, 60, 44, 22, 16,
        'LSTM Decoder\nctx ⊕ embed → LSTM\nh = 512',
        *C_MODEL, bold=True, fontsize=10)
    arrow(ax, 54, 52, 60, 52)

    # Beam search
    box(ax, 86, 44, 12, 16,
        'Beam search\nB = 5\nTop-K out',
        *C_OUTPUT, fontsize=9)
    arrow(ax, 82, 52, 86, 52)

    # Loop arrow on decoder
    arrow(ax, 71, 44, 71, 38, style='-|>')
    arrow(ax, 71, 38, 71, 32, style='-')
    label(ax, 71, 35, 'token tᵤ → tᵤ₊₁ (teacher forcing)', fontsize=8, color='#5F6368')

    # Cached input
    arrow(ax, 92, 76, 92, 64, style='-')
    arrow(ax, 92, 64, 15, 64, style='-')
    arrow(ax, 15, 64, 15, 60)

    # Output box
    box(ax, 30, 12, 40, 14,
        'Token sequence prediction\n(<sos> w₁ w₂ … wₙ <eos>)',
        *C_OUTPUT, bold=True, fontsize=10)
    arrow(ax, 50, 44, 50, 26)

    # Training notes
    box(ax, 4, 12, 22, 14,
        'Training\nAdam · CE (ignore <pad>)\n200 epochs · early stop\nteacher forcing 50%↓',
        *C_NOTE, fontsize=9)
    box(ax, 74, 12, 24, 14,
        'Inference\n~18 ms / video (GPU)\nfrom cached features',
        *C_NOTE, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'lstm_seq2seq_pipeline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'wrote {out}')

# =============================================================================
# 4. TRANSFORMER PIPELINE (ViT + encoder-decoder)
# =============================================================================
def make_transformer():
    fig, ax = setup_axes((13, 8))
    title(ax, 'Approach 4 — Transformer (ViT-Small frame encoder + temporal Encoder–Decoder)')

    # Frame sampling
    box(ax, 2,  78, 12, 10, 'Video\n(.mp4)', *C_INPUT, bold=True)
    box(ax, 16, 78, 18, 10, 'Uniform sampling\n16 frames · 224×224', *C_PREPROC, fontsize=9)
    box(ax, 36, 78, 22, 10,
        'Augmentation (train)\nresized crop · flip · color jitter\nCutOut · temporal jitter / reverse',
        *C_PREPROC, fontsize=8)

    arrow(ax, 14, 83, 16, 83)
    arrow(ax, 34, 83, 36, 83)

    # ViT frame encoder
    box(ax, 60, 78, 22, 10,
        'ViT-Small/16 (timm)\nCLS token → 384-D / frame\nfrozen for first 5 epochs',
        *C_FEATURE, bold=True, fontsize=9)
    arrow(ax, 58, 83, 60, 83)

    # Project + pos enc
    box(ax, 84, 78, 14, 10,
        'Linear → d=256\n+ sinusoidal\npos. encoding',
        *C_FEATURE, fontsize=9)
    arrow(ax, 82, 83, 84, 83)

    # Drop down
    arrow(ax, 91, 78, 91, 64)

    # Temporal Encoder
    box(ax, 60, 50, 32, 14,
        'Temporal Transformer Encoder\n4 layers · d=256 · heads=8 · FFN=1024\nself-attention over 16 frame tokens',
        *C_MODEL, bold=True, fontsize=9)
    arrow(ax, 76, 50, 76, 40)

    # Cross-attention arrow into decoder
    box(ax, 32, 26, 32, 14,
        'Transformer Decoder\n4 layers · d=256 · heads=8\ncausal mask + cross-attention\nlabel smoothing 0.1',
        *C_MODEL, bold=True, fontsize=9)
    arrow(ax, 60, 33, 64, 33)

    # Tokens loop
    box(ax, 4, 26, 24, 14,
        'Token embedding\n+ pos. encoding\nteacher forcing (train)',
        *C_FEATURE, fontsize=9)
    arrow(ax, 28, 33, 32, 33)

    # Output projection
    box(ax, 70, 22, 24, 12,
        'Linear → vocab logits\nargmax / beam search',
        *C_OUTPUT, fontsize=10)
    arrow(ax, 64, 26, 70, 26)

    # Final output
    box(ax, 30, 6, 40, 12,
        'Token sequence prediction\n(<sos> w₁ w₂ … wₙ <eos>)',
        *C_OUTPUT, bold=True, fontsize=10)
    arrow(ax, 82, 22, 50, 18)

    # Training notes
    label(ax, 50, -2,
          'AMP · backbone LR 1e-5 / head LR 1e-4 · warm-up 300 + cosine decay · grad-clip 1.0',
          fontsize=9, color='#5F6368')

    ax.set_ylim(-6, 100)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'transformer_pipeline.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'wrote {out}')

if __name__ == '__main__':
    make_cosine()
    make_dtw()
    make_lstm()
    make_transformer()
    print('done.')

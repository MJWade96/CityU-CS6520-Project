"""
Medical RAG System Architecture Diagram
Systematic Cartography Style
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np

# ── Canvas Setup ──
fig, ax = plt.subplots(1, 1, figsize=(32, 44), dpi=200)
fig.patch.set_facecolor('#FAFAF8')
ax.set_facecolor('#FAFAF8')
ax.set_xlim(0, 100)
ax.set_ylim(0, 140)
ax.axis('off')

# ── Color Palette ──
C_BG = '#FAFAF8'
C_DARK = '#1A1A1A'
C_MUTED = '#6B6B6B'
C_ACCENT_WARM = '#C4553A'       # Generation stage
C_ACCENT_COOL = '#2E6B8A'       # Retrieval stage
C_ACCENT_TEAL = '#3A8A7C'       # Reranking stage
C_ACCENT_AMBER = '#B8860B'      # Query rewrite
C_ACCENT_PURPLE = '#6B4C8A'     # Indexing / data
C_ACCENT_GRAY = '#8A8A8A'       # Evaluation
C_MODULE_BG = '#FFFFFF'
C_MODULE_BORDER = '#D0D0D0'
C_PARAM_BG = '#F0EDE8'
C_HEADER_WARM = '#C4553A'
C_HEADER_COOL = '#2E6B8A'
C_HEADER_TEAL = '#3A8A7C'
C_HEADER_AMBER = '#B8860B'
C_HEADER_PURPLE = '#6B4C8A'
C_HEADER_GRAY = '#8A8A8A'
C_ARROW = '#999999'
C_CONNECTOR = '#B0B0B0'

FONTS_DIR = r'C:\Users\MJWade\.codebuddy\skills\canvas-design\canvas-fonts'

import matplotlib.font_manager as fm
fm.fontManager.addfont(f'{FONTS_DIR}/GeistMono-Regular.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/GeistMono-Bold.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/InstrumentSans-Bold.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/InstrumentSans-Regular.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/Jura-Light.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/WorkSans-Bold.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/WorkSans-Regular.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/Jura-Medium.ttf')
fm.fontManager.addfont(f'{FONTS_DIR}/DMMono-Regular.ttf')

F_TITLE = fm.FontProperties(fname=f'{FONTS_DIR}/InstrumentSans-Bold.ttf', size=28)
F_SECTION = fm.FontProperties(fname=f'{FONTS_DIR}/InstrumentSans-Bold.ttf', size=13)
F_MODULE = fm.FontProperties(fname=f'{FONTS_DIR}/GeistMono-Bold.ttf', size=8.5)
F_PARAM = fm.FontProperties(fname=f'{FONTS_DIR}/GeistMono-Regular.ttf', size=6.8)
F_PARAM_LABEL = fm.FontProperties(fname=f'{FONTS_DIR}/GeistMono-Regular.ttf', size=6.2)
F_LABEL = fm.FontProperties(fname=f'{FONTS_DIR}/InstrumentSans-Regular.ttf', size=7.5)
F_SMALL = fm.FontProperties(fname=f'{FONTS_DIR}/GeistMono-Regular.ttf', size=5.8)
F_PHASE = fm.FontProperties(fname=f'{FONTS_DIR}/WorkSans-Bold.ttf', size=10)
F_SUBTITLE = fm.FontProperties(fname=f'{FONTS_DIR}/InstrumentSans-Regular.ttf', size=10)
F_HEADER = fm.FontProperties(fname=f'{FONTS_DIR}/GeistMono-Bold.ttf', size=7.5)
F_STEP = fm.FontProperties(fname=f'{FONTS_DIR}/Jura-Medium.ttf', size=8.5)
F_FOOTNOTE = fm.FontProperties(fname=f'{FONTS_DIR}/DMMono-Regular.ttf', size=5.5)
F_BIG_NUM = fm.FontProperties(fname=f'{FONTS_DIR}/InstrumentSans-Bold.ttf', size=48)

# ── Helper Functions ──
def draw_module_box(ax, x, y, w, h, title, params, header_color, bg_color='#FFFFFF', border_color=None):
    """Draw a module box with header and parameters."""
    if border_color is None:
        border_color = header_color
    
    # Main box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=0.8,
        alpha=0.95,
        zorder=2
    )
    ax.add_patch(box)
    
    # Header bar
    header = FancyBboxPatch(
        (x + 0.3, y + h - 2.2), w - 0.6, 2.0,
        boxstyle="round,pad=0.15",
        facecolor=header_color,
        edgecolor='none',
        alpha=0.9,
        zorder=3
    )
    ax.add_patch(header)
    
    # Title
    ax.text(x + w/2, y + h - 1.2, title,
            fontproperties=F_MODULE, color='white',
            ha='center', va='center', zorder=4)
    
    # Parameters
    py = y + h - 3.6
    for label, value in params:
        # Label
        ax.text(x + 1.2, py, label + ':',
                fontproperties=F_PARAM_LABEL, color=C_MUTED,
                ha='left', va='center', zorder=4)
        # Value
        ax.text(x + w - 1.0, py, value,
                fontproperties=F_PARAM, color=C_DARK,
                ha='right', va='center', zorder=4)
        # Separator line
        ax.plot([x + 1.0, x + w - 0.8], [py - 0.6, py - 0.6],
                color='#E8E5E0', linewidth=0.3, zorder=3)
        py -= 1.35
    
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style='->', connectionstyle="arc3,rad=0"):
    """Draw a connector arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        mutation_scale=12,
        zorder=5
    )
    ax.add_patch(arrow)

def draw_phase_indicator(ax, x, y, text, color):
    """Draw a phase number indicator."""
    circle = plt.Circle((x, y), 0.8, color=color, alpha=0.85, zorder=6)
    ax.add_patch(circle)

def draw_param_annotation(ax, x, y, text, anchor='left'):
    """Draw a floating parameter annotation."""
    ha = 'left' if anchor == 'left' else 'right'
    bbox_props = dict(boxstyle='round,pad=0.2', facecolor=C_PARAM_BG, edgecolor='#D0CFC8', linewidth=0.4)
    ax.text(x, y, text, fontproperties=F_SMALL, color=C_MUTED,
            ha=ha, va='center', bbox=bbox_props, zorder=4)

def draw_flow_line(ax, points, color=C_CONNECTOR, lw=0.8, style='-'):
    """Draw a flow line through multiple points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style, zorder=1, solid_capstyle='round')

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(50, 137.5, 'MEDICAL RAG SYSTEM', fontproperties=F_TITLE, color=C_DARK,
        ha='center', va='center', zorder=10)
ax.text(50, 135.5, 'Enhanced Retrieval-Augmented Generation for Clinical Question Answering',
        fontproperties=F_SUBTITLE, color=C_MUTED, ha='center', va='center', zorder=10)

# Decorative line under title
ax.plot([15, 85], [134.8, 134.8], color=C_DARK, linewidth=0.8, zorder=10)
ax.plot([30, 70], [134.3, 134.3], color=C_MUTED, linewidth=0.3, zorder=10)

# ══════════════════════════════════════════════════════════════
# PHASE LABELS (left sidebar)
# ══════════════════════════════════════════════════════════════
phases = [
    (5, 129, '0', 'DATA', C_ACCENT_PURPLE),
    (5, 111.5, '1', 'QUERY', C_ACCENT_AMBER),
    (5, 88, '2', 'RETRIEVE', C_ACCENT_COOL),
    (5, 64, '3', 'RERANK', C_ACCENT_TEAL),
    (5, 46, '4', 'GENERATE', C_ACCENT_WARM),
    (5, 28, '5', 'EVALUATE', C_ACCENT_GRAY),
]

for px, py, num, label, color in phases:
    draw_phase_indicator(ax, px, py, num, color)
    ax.text(px, py, num, fontproperties=fm.FontProperties(fname=f'{FONTS_DIR}/WorkSans-Bold.ttf', size=12),
            color='white', ha='center', va='center', zorder=7)
    ax.text(px + 1.8, py, label, fontproperties=F_PHASE, color=color,
            ha='left', va='center', zorder=7)
    # Phase divider line
    if py < 130:
        ax.plot([9.5, 95], [py + 5.2, py + 5.2], color='#E8E5E0', linewidth=0.4, zorder=1)

# Vertical line for phases
ax.plot([7.5, 7.5], [28, 133], color='#E0DDD8', linewidth=0.6, zorder=1)

# ══════════════════════════════════════════════════════════════
# PHASE 0: DATA & INDEXING (y=121..133)
# ══════════════════════════════════════════════════════════════

# Corpora Sources
draw_module_box(ax, 12, 125, 24, 7, 'CORPUS SOURCES', [
    ('textbooks', 'medical textbooks'),
    ('pubmed', 'PubMed abstracts'),
    ('statpearls', 'StatPearls articles'),
], C_ACCENT_PURPLE)

# combined_corpus.json
draw_module_box(ax, 12, 118, 24, 6, 'COMBINED CORPUS', [
    ('format', 'JSON array'),
    ('fields', 'content + metadata'),
    ('sources', 'textbooks + pubmed + statpearls'),
], C_ACCENT_PURPLE, bg_color='#F8F6F3')

# Vector Index Build
draw_module_box(ax, 42, 118, 28, 13, 'VECTOR INDEX BUILDER', [
    ('embedding_model', 'BAAI/bge-m3 (default)'),
    ('normalize_embeddings', 'True'),
    ('batch_size', '256'),
    ('store_type', 'FAISS'),
    ('distance_strategy', 'COSINE'),
    ('output', 'faiss_index/ + build_metadata.json'),
], C_ACCENT_PURPLE, bg_color='#F8F6F3')

# Chunking options (right side)
draw_module_box(ax, 76, 125, 18, 7, 'CHUNKING', [
    ('SemanticChunker', 'size=512, overlap=50'),
    ('ParentChild', 'parent=1024, child=200'),
    ('SlidingWindow', 'size=512, overlap=50'),
], C_ACCENT_PURPLE, bg_color='#F5F2ED')

# Metadata Enhancement
draw_module_box(ax, 76, 118, 18, 6, 'METADATA', [
    ('LLM generator', 'summary + keywords'),
    ('Rule-based', 'TF-IDF + regex entities'),
], C_ACCENT_PURPLE, bg_color='#F5F2ED')

# Arrows
draw_arrow(ax, 36, 128.5, 42, 128.5, C_ACCENT_PURPLE)
draw_arrow(ax, 36, 121, 42, 121, C_ACCENT_PURPLE)
draw_arrow(ax, 70, 128.5, 76, 128.5, C_ACCENT_PURPLE, style='->')
draw_arrow(ax, 70, 121, 76, 121, C_ACCENT_PURPLE, style='->')
draw_arrow(ax, 85, 125, 85, 124.2, C_ACCENT_PURPLE)

# ══════════════════════════════════════════════════════════════
# PHASE 1: QUERY REWRITING (y=94..116)
# ══════════════════════════════════════════════════════════════

# Input query
draw_module_box(ax, 12, 106, 20, 8, 'INPUT QUERY', [
    ('source', 'MedQA dataset'),
    ('format', '{question, options, answer}'),
    ('split', 'dev=300, test=rest'),
], C_ACCENT_AMBER, bg_color='#FFFAF0')

# Dictionary Rewriter
draw_module_box(ax, 38, 106, 24, 8, 'DICT REWRITER', [
    ('abbreviations', '14 mappings (mi→myocardial...)'),
    ('synonyms', '21 mappings (heart attack→...)'),
    ('chinese_terms', '10 mappings (心梗→心肌梗死...)'),
    ('strategy', 'expand → replace → chinese'),
], C_ACCENT_AMBER, bg_color='#FFFAF0')

# LLM Query Rewriter
draw_module_box(ax, 38, 95, 24, 10, 'LLM QUERY REWRITER', [
    ('provider', 'Qwen3-4B (shared API)'),
    ('temperature', '0.1'),
    ('max_tokens', '200'),
    ('enable_thinking', 'False'),
    ('auto_mode', 'auto (max 160 chars / 24 words)'),
    ('prompt', 'expand terms, remove ambiguity'),
], C_ACCENT_AMBER, bg_color='#FFFAF0')

# Query Expansion (optional)
draw_module_box(ax, 68, 100, 24, 8, 'QUERY EXPANDER', [
    ('status', 'DISABLED (use_expansion=False)'),
    ('num_expansions', '3'),
    ('temperature', '0.3'),
    ('max_tokens', '300'),
    ('strategy', 'Multi-Query (diverse angles)'),
], C_ACCENT_AMBER, bg_color='#F5F2ED', border_color='#D0CFC8')

# Arrow for "disabled"
ax.text(80, 108.5, 'not active', fontproperties=F_SMALL, color='#C0B8A8',
        ha='center', va='center', style='italic', zorder=4)

# Rewrite Pipeline wrapper
rewrite_pipeline_box = FancyBboxPatch(
    (36, 94), 58, 21,
    boxstyle="round,pad=0.4",
    facecolor='none',
    edgecolor=C_ACCENT_AMBER,
    linewidth=1.2,
    linestyle='--',
    alpha=0.5,
    zorder=1
)
ax.add_patch(rewrite_pipeline_box)
ax.text(94, 114.5, 'QueryRewritePipeline', fontproperties=F_SMALL, color=C_ACCENT_AMBER,
        ha='right', va='center', style='italic', zorder=4)

# Arrows
draw_arrow(ax, 32, 110, 38, 110, C_ACCENT_AMBER)
draw_arrow(ax, 50, 106, 50, 105, C_ACCENT_AMBER)
draw_arrow(ax, 62, 100, 68, 104, C_ACCENT_AMBER, style='->', connectionstyle="arc3,rad=-0.15")
ax.text(66, 98, 'optional', fontproperties=F_SMALL, color='#C0B8A8', ha='center', va='center',
        style='italic', zorder=4)

# Primary query output
draw_flow_line(ax, [(50, 95), (50, 92)], C_ACCENT_AMBER, lw=1.5)
ax.text(51, 93.2, 'primary_query', fontproperties=F_SMALL, color=C_ACCENT_AMBER,
        ha='left', va='center', zorder=4)

# ══════════════════════════════════════════════════════════════
# PHASE 2: HYBRID RETRIEVAL (y=70..93)
# ══════════════════════════════════════════════════════════════

# FAISS Dense Retriever
draw_module_box(ax, 12, 82, 26, 11, 'DENSE RETRIEVAL', [
    ('backend', 'FAISS (persisted index)'),
    ('embedding', 'BAAI/bge-m3 (from metadata)'),
    ('normalize', 'True'),
    ('device', 'auto (cuda → mps → cpu)'),
    ('fetch_k', 'k × 3'),
], C_ACCENT_COOL, bg_color='#F0F6FA')

# BM25 Sparse Retriever
draw_module_box(ax, 44, 82, 26, 11, 'BM25 SPARSE RETRIEVAL', [
    ('tokenizer', 'regex \\b\\w+\\b (lowercase)'),
    ('k1', '1.5'),
    ('b', '0.75'),
    ('cache', 'pickle (fingerprint-based)'),
    ('fetch_k', 'k × 3'),
], C_ACCENT_COOL, bg_color='#F0F6FA')

# RRF Fusion
draw_module_box(ax, 76, 82, 18, 11, 'RRF FUSION', [
    ('rrf_k', '60 (standard)'),
    ('dense_weight', '0.5'),
    ('sparse_weight', '0.5'),
    ('formula', 'α·1/(k+rank_d) +'),
    ('', '(1-α)·1/(k+rank_s)'),
], C_ACCENT_COOL, bg_color='#F0F6FA')

# Hybrid Retriever wrapper
hybrid_box = FancyBboxPatch(
    (10, 70), 86, 24,
    boxstyle="round,pad=0.4",
    facecolor='none',
    edgecolor=C_ACCENT_COOL,
    linewidth=1.2,
    linestyle='--',
    alpha=0.5,
    zorder=1
)
ax.add_patch(hybrid_box)
ax.text(95.5, 93.5, 'HybridRetriever', fontproperties=F_SMALL, color=C_ACCENT_COOL,
        ha='right', va='center', style='italic', zorder=4)

# Adaptive Retriever note
draw_param_annotation(ax, 12, 70, 'AdaptiveRetriever: rule-based skip (greetings, short queries) — DISABLED')

# Arrows for dense + sparse into fusion
draw_arrow(ax, 38, 87.5, 44, 87.5, C_ACCENT_COOL, lw=1.5)
draw_arrow(ax, 70, 87.5, 76, 87.5, C_ACCENT_COOL, lw=1.5)

# Output
draw_flow_line(ax, [(85, 82), (85, 78), (50, 78)], C_ACCENT_COOL, lw=1.5)
ax.text(51, 78.5, f'k×2 candidates (fetch_k=15)', fontproperties=F_SMALL, color=C_ACCENT_COOL,
        ha='left', va='center', zorder=4)

# Query input arrow
draw_flow_line(ax, [(50, 92), (25, 92), (25, 93)], C_ACCENT_AMBER, lw=1.2)
draw_flow_line(ax, [(50, 92), (57, 92), (57, 93)], C_ACCENT_AMBER, lw=1.2)

# ══════════════════════════════════════════════════════════════
# PHASE 3: RERANKING (y=56..69)
# ══════════════════════════════════════════════════════════════

# Cross-Encoder
draw_module_box(ax, 12, 58, 28, 10, 'CROSS-ENCODER RERANKER', [
    ('model', 'BAAI/bge-reranker-large'),
    ('device', 'auto'),
    ('scoring', 'CrossEncoder(query, doc)'),
    ('output_top_k', 'top_k × 2 = 10'),
    ('fallback', 'original order if unavailable'),
], C_ACCENT_TEAL, bg_color='#F0FAF8')

# MMR (disabled)
draw_module_box(ax, 46, 58, 22, 7, 'MMR DIVERSITY', [
    ('status', 'DISABLED'),
    ('lambda', '0.5'),
    ('similarity', 'Jaccard index'),
], C_ACCENT_TEAL, bg_color='#F5F2ED', border_color='#D0CFC8')
ax.text(57, 58.3, 'not active', fontproperties=F_SMALL, color='#B0D0C8',
        ha='center', va='center', style='italic', zorder=4)

# Lost-in-Middle (disabled)
draw_module_box(ax, 46, 49, 22, 7, 'LOST-IN-MIDDLE', [
    ('status', 'DISABLED'),
    ('strategy', 'relevance →首尾交替'),
], C_ACCENT_TEAL, bg_color='#F5F2ED', border_color='#D0CFC8')
ax.text(57, 49.3, 'not active', fontproperties=F_SMALL, color='#B0D0C8',
        ha='center', va='center', style='italic', zorder=4)

# Reranker Pipeline wrapper
reranker_box = FancyBboxPatch(
    (10, 48), 60, 21,
    boxstyle="round,pad=0.4",
    facecolor='none',
    edgecolor=C_ACCENT_TEAL,
    linewidth=1.2,
    linestyle='--',
    alpha=0.5,
    zorder=1
)
ax.add_patch(reranker_box)
ax.text(69.5, 68.5, 'RerankerPipeline', fontproperties=F_SMALL, color=C_ACCENT_TEAL,
        ha='right', va='center', style='italic', zorder=4)

# Reranker config summary box
draw_module_box(ax, 76, 58, 18, 10, 'RERANK CONFIG', [
    ('use_cross_encoder', 'True'),
    ('use_mmr', 'False'),
    ('use_lost_in_middle', 'False'),
    ('final_top_k', '5'),
    ('pipeline', 'CE → [MMR] → [LiM]'),
], C_ACCENT_TEAL, bg_color='#F0FAF8')

# Arrows
draw_arrow(ax, 40, 63, 46, 62, C_ACCENT_TEAL, style='->', connectionstyle="arc3,rad=0.1")
draw_arrow(ax, 68, 63, 76, 63, C_ACCENT_TEAL)

# Input from retrieval
draw_flow_line(ax, [(50, 78), (50, 75), (26, 75), (26, 68)], C_ACCENT_COOL, lw=1.2)
draw_flow_line(ax, [(85, 78), (85, 68), (40, 68)], C_ACCENT_COOL, lw=1.0, style='--')

# Output
draw_flow_line(ax, [(85, 63), (92, 63), (92, 48), (50, 48)], C_ACCENT_TEAL, lw=1.5)
ax.text(51, 48.5, 'top_k=5 documents', fontproperties=F_SMALL, color=C_ACCENT_TEAL,
        ha='left', va='center', zorder=4)

# ══════════════════════════════════════════════════════════════
# PHASE 4: GENERATION (y=34..47)
# ══════════════════════════════════════════════════════════════

# Context formatting
draw_module_box(ax, 12, 37, 24, 10, 'CONTEXT FORMATTER', [
    ('format', '"[{i}] {context}"'),
    ('separator', '"\\n\\n"'),
    ('sources', 'top-5 reranked docs'),
], C_ACCENT_WARM, bg_color='#FFFAF0')

# Prompt Builder
draw_module_box(ax, 42, 37, 26, 10, 'MEDICAL EVAL PROMPT', [
    ('system', 'medical expert assistant'),
    ('instruction', 'answer based on context'),
    ('fallback', 'insufficient info → state so'),
    ('output_format', 'Answer: [A/B/C/D/E]'),
    ('CoT mode', 'DISABLED'),
], C_ACCENT_WARM, bg_color='#FFFAF0')

# LLM Generator
draw_module_box(ax, 74, 37, 20, 10, 'LLM GENERATOR', [
    ('provider', 'Qwen3-4B'),
    ('base_url', 'wishub-x6.ctyun.cn/v1'),
    ('temperature', '0.1'),
    ('timeout', '120.0s'),
    ('max_retries', '5 (exp. backoff)'),
    ('enable_thinking', 'False'),
], C_ACCENT_WARM, bg_color='#FFFAF0')

# Answer extraction
draw_module_box(ax, 42, 28, 26, 8, 'ANSWER EXTRACTION', [
    ('patterns', 'Answer: [X], **(X)**, [(X)]'),
    ('fallback', 'last standalone A-E letter'),
    ('output', 'predicted_answer (str)'),
], C_ACCENT_WARM, bg_color='#F8F4F0')

# Async generation wrapper
async_box = FancyBboxPatch(
    (40, 27), 56, 21,
    boxstyle="round,pad=0.4",
    facecolor='none',
    edgecolor=C_ACCENT_WARM,
    linewidth=1.2,
    linestyle='--',
    alpha=0.5,
    zorder=1
)
ax.add_patch(async_box)
ax.text(95.5, 47.5, 'EnhancedMedicalLLMGenerator', fontproperties=F_SMALL, color=C_ACCENT_WARM,
        ha='right', va='center', style='italic', zorder=4)

# Concurrency note
draw_param_annotation(ax, 74, 28, 'async: Semaphore(4) + RateLimiter(0.9 rps, burst=4)')

# Arrows
draw_arrow(ax, 36, 42, 42, 42, C_ACCENT_WARM, lw=1.5)
draw_arrow(ax, 68, 42, 74, 42, C_ACCENT_WARM, lw=1.5)
draw_arrow(ax, 84, 37, 92, 37, C_ACCENT_WARM, style='->')
draw_flow_line(ax, [(92, 37), (92, 32), (68, 32)], C_ACCENT_WARM, lw=1.2)
draw_arrow(ax, 68, 32, 68, 36, C_ACCENT_WARM)

# Input from reranking
draw_flow_line(ax, [(50, 48), (50, 47), (24, 47), (24, 47)], C_ACCENT_TEAL, lw=1.2)

# ══════════════════════════════════════════════════════════════
# PHASE 5: EVALUATION (y=16..27)
# ══════════════════════════════════════════════════════════════

# Evaluation framework
draw_module_box(ax, 12, 17, 30, 10, 'EVALUATION FRAMEWORK', [
    ('dataset', 'MedQA (USMLE-style)'),
    ('dev_set', '300 questions (not evaluated)'),
    ('test_set', 'remaining questions'),
    ('metric', 'accuracy = correct / total'),
    ('comparison', 'McNemar test (p=0.05)'),
], C_ACCENT_GRAY, bg_color='#F5F5F3')

# Concurrency config
draw_module_box(ax, 48, 17, 24, 10, 'CONCURRENCY CONFIG', [
    ('max_concurrent', '4'),
    ('rpm_limit', '60'),
    ('in_flight_multiplier', '2'),
    ('save_every', '5 questions'),
    ('heartbeat', '15s interval'),
], C_ACCENT_GRAY, bg_color='#F5F5F3')

# Progress & resilience
draw_module_box(ax, 78, 17, 16, 10, 'RESILIENCE', [
    ('checkpoint', 'checkpoint.json'),
    ('resume', 'auto (stale check)'),
    ('retry', 'exp. backoff 1×2^n'),
    ('heartbeat_log', 'enabled'),
], C_ACCENT_GRAY, bg_color='#F5F5F3')

# Arrows from generation
draw_flow_line(ax, [(42, 32), (42, 27)], C_ACCENT_WARM, lw=1.2)
draw_arrow(ax, 42, 22, 48, 22, C_ACCENT_GRAY)

# ══════════════════════════════════════════════════════════════
# MAIN FLOW CONNECTOR (right side vertical arrow)
# ══════════════════════════════════════════════════════════════
# Big vertical flow on the far right
flow_points = [
    (96, 128), (96, 110), (96, 87), (96, 63), (96, 42), (96, 22)
]
for i in range(len(flow_points) - 1):
    ax.annotate('', xy=flow_points[i+1], xytext=flow_points[i],
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8,
                               connectionstyle='arc3,rad=0'),
                zorder=6)

# Flow labels on right
flow_labels = [
    (96, 119, 'rewrite'),
    (96, 98, 'retrieve'),
    (96, 75, 'rerank'),
    (96, 52, 'generate'),
    (96, 32, 'evaluate'),
]
for fx, fy, ft in flow_labels:
    ax.text(fx + 1.2, fy, ft, fontproperties=F_SMALL, color=C_DARK,
            ha='left', va='center', zorder=7, alpha=0.6)

# ══════════════════════════════════════════════════════════════
# DECISION PARAMETERS (bottom strip)
# ══════════════════════════════════════════════════════════════
param_strip_y = 10
ax.plot([10, 95], [param_strip_y + 4, param_strip_y + 4], color='#E0DDD8', linewidth=0.6, zorder=1)
ax.text(50, param_strip_y + 6.5, 'DECISION DIMENSIONS', fontproperties=F_SECTION, color=C_DARK,
        ha='center', va='center', zorder=10)

# Decision dimension items
dimensions = [
    ('DATA', 'corpus sources · chunk_size=512 · overlap=50 · parent=1024 · child=200', C_ACCENT_PURPLE),
    ('RETRIEVAL', 'embedding=BAAI/bge-m3 · FAISS+COSINE · BM25(k1=1.5, b=0.75) · RRF(k=60, α=0.5)', C_ACCENT_COOL),
    ('RERANK', 'CE=bge-reranker-large · top_k=5 · MMR(λ=0.5) · Lost-in-Middle', C_ACCENT_TEAL),
    ('GENERATION', 'Qwen3-4B · temp=0.1 · no thinking · structured output A-E', C_ACCENT_WARM),
    ('EVALUATION', 'MedQA · dev=300 · McNemar p=0.05 · Bootstrap CI', C_ACCENT_GRAY),
]

for i, (dlabel, ddesc, dcolor) in enumerate(dimensions):
    dy = param_strip_y - i * 2.2
    # Color dot
    ax.plot(12, dy, 'o', color=dcolor, markersize=4, zorder=10)
    ax.text(13.5, dy, dlabel, fontproperties=F_PARAM_LABEL, color=dcolor,
            ha='left', va='center', zorder=10)
    ax.text(24, dy, ddesc, fontproperties=F_FOOTNOTE, color=C_MUTED,
            ha='left', va='center', zorder=10)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
ax.text(50, 0.5, 'enhanced_eval.py · EnhancedRAGPipeline · python-rag/',
        fontproperties=F_FOOTNOTE, color='#B0B0B0', ha='center', va='center', zorder=10)

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
plt.savefig(
    r'c:\Users\MJWade\AppData\Roaming\CodeBuddy CN\User\globalStorage\tencent-cloud.coding-copilot\brain\4d401b823b8440d6b873554ec9093f9e\rag_architecture.pdf',
    format='pdf',
    dpi=200,
    bbox_inches='tight',
    facecolor=C_BG,
    edgecolor='none'
)
plt.savefig(
    r'c:\Users\MJWade\AppData\Roaming\CodeBuddy CN\User\globalStorage\tencent-cloud.coding-copilot\brain\4d401b823b8440d6b873554ec9093f9e\rag_architecture.png',
    format='png',
    dpi=200,
    bbox_inches='tight',
    facecolor=C_BG,
    edgecolor='none'
)
print("Architecture diagram saved successfully.")
plt.close()

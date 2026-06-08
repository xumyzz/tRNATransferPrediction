#!/usr/bin/env python3
"""
单条嵌合体 contact map 可视化

输出 HTML 文件，可直接在浏览器打开。
用法:
  python viz_single.py --npy ./chimera_predictions/bpRNA_CRW_15563.npy --out contact.html
"""

import numpy as np
import argparse
import os
import json
import base64


def cmap_to_html(cmap, seq_hint="", trna_5end=43, trna_3end=15, title=""):
    """将 contact map 渲染为独立 HTML 文件"""

    L = cmap.shape[0]
    ins_start = trna_5end
    ins_end = L - trna_3end

    pairs = list(zip(*np.where(np.triu(cmap) > 0)))
    cross_5_3 = [(i, j) for i, j in pairs
                 if i < ins_start and j >= ins_end]
    t5_pairs = [(i, j) for i, j in pairs
                if i < ins_start and j < ins_start]
    ins_pairs = [(i, j) for i, j in pairs
                 if i >= ins_start and i < ins_end and j >= ins_start and j < ins_end]
    leak = [(i, j) for i, j in pairs
            if (i < ins_start and ins_start <= j < ins_end) or
               (i >= ins_end and ins_start <= j < ins_end)]

    # ── 构建 HTML heatmap 数据 ──
    cells_js = []
    for i in range(L):
        for j in range(L):
            if cmap[i, j] > 0:
                cells_js.append([i, j, 1])

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Contact Map - {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
.container {{ max-width: 900px; margin: 0 auto; }}
.header {{ background: #2c3e50; color: white; padding: 20px 30px; border-radius: 8px 8px 0 0; }}
.header h1 {{ font-size: 20px; margin-bottom: 8px; }}
.header .meta {{ font-size: 13px; opacity: 0.85; }}
.content {{ background: white; padding: 30px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
canvas {{ border: 1px solid #ddd; border-radius: 4px; display: block; margin: 0 auto; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px; }}
.stat-card {{ background: #f8f9fa; border-radius: 6px; padding: 15px; text-align: center; }}
.stat-card .value {{ font-size: 28px; font-weight: 700; color: #2c3e50; }}
.stat-card .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.stat-card.green .value {{ color: #27ae60; }}
.stat-card.orange .value {{ color: #e67e22; }}
.stat-card.red .value {{ color: #e74c3c; }}
.pair-list {{ margin-top: 20px; }}
.pair-list h3 {{ font-size: 14px; color: #555; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
.pair-group {{ margin-bottom: 8px; font-size: 13px; font-family: 'Consolas', monospace; }}
.pair-group .tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-right: 6px; font-weight: 600; }}
.tag-cross {{ background: #e67e22; color: white; }}
.tag-t5 {{ background: #3498db; color: white; }}
.tag-ins {{ background: #2ecc71; color: white; }}
.tag-leak {{ background: #e74c3c; color: white; }}
.pair-group .items {{ color: #555; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>{title or 'Contact Map'}</h1>
  <div class="meta">{L} nt &nbsp;|&nbsp;
       5'={trna_5end}nt &nbsp;|&nbsp;
       insert={ins_end - ins_start}nt &nbsp;|&nbsp;
       3'={trna_3end}nt &nbsp;|&nbsp;
       {len(pairs)} pairs</div>
</div>
<div class="content">
  <canvas id="cmap" width="800" height="800"></canvas>
  <div class="stats">
    <div class="stat-card">
      <div class="value">{len(pairs)}</div>
      <div class="label">Total Pairs</div>
    </div>
    <div class="stat-card green">
      <div class="value">{len(t5_pairs)}</div>
      <div class="label">5' tRNA internal</div>
    </div>
    <div class="stat-card green">
      <div class="value">{len(ins_pairs)}</div>
      <div class="label">Insert internal</div>
    </div>
    <div class="stat-card orange">
      <div class="value">{len(cross_5_3)}</div>
      <div class="label">5'↔3' Cross</div>
    </div>
  </div>
  <div class="pair-list">
    <h3>5'↔3' Cross Pairs ({len(cross_5_3)})</h3>
    <div class="pair-group">
      <span class="tag tag-cross">CROSS</span>
      <span class="items">{', '.join(f'({i},{j})' for i, j in cross_5_3[:30])}{' ...' if len(cross_5_3) > 30 else ''}</span>
    </div>"""

    if t5_pairs:
        html += f"""
    <h3>5' tRNA Internal ({len(t5_pairs)})</h3>
    <div class="pair-group">
      <span class="tag tag-t5">T5</span>
      <span class="items">{', '.join(f'({i},{j})' for i, j in t5_pairs[:20])}{' ...' if len(t5_pairs) > 20 else ''}</span>
    </div>"""

    if leak:
        html += f"""
    <h3 style="color:#e74c3c">⚠ Cross-domain Leak ({len(leak)})</h3>
    <div class="pair-group">
      <span class="tag tag-leak">LEAK</span>
      <span class="items">{', '.join(f'({i},{j})' for i, j in leak)}</span>
    </div>"""

    html += f"""
  </div>
</div>
</div>
<script>
const canvas = document.getElementById('cmap');
const ctx = canvas.getContext('2d');
const L = {L};
const insS = {ins_start};
const insE = {ins_end};

// 背景
ctx.fillStyle = '#f8f9fa';
ctx.fillRect(0, 0, 800, 800);

// 分隔线
ctx.strokeStyle = '#e74c3c';
ctx.lineWidth = 2;
ctx.setLineDash([6, 4]);
ctx.beginPath();
ctx.moveTo(0, insS / L * 800);
ctx.lineTo(800, insS / L * 800);
ctx.moveTo(0, insE / L * 800);
ctx.lineTo(800, insE / L * 800);
ctx.moveTo(insS / L * 800, 0);
ctx.lineTo(insS / L * 800, 800);
ctx.moveTo(insE / L * 800, 0);
ctx.lineTo(insE / L * 800, 800);
ctx.stroke();
ctx.setLineDash([]);

// 配对点
const cells = {json.dumps(cells_js)};
const cs = 800 / L;
ctx.fillStyle = '#2c3e50';
for (const [i, j, v] of cells) {{
    ctx.fillRect(j * cs, i * cs, Math.max(cs, 2), Math.max(cs, 2));
    ctx.fillRect(i * cs, j * cs, Math.max(cs, 2), Math.max(cs, 2));
}}

// 对角线
ctx.strokeStyle = '#ddd';
ctx.lineWidth = 1;
ctx.beginPath();
ctx.moveTo(0, 0);
ctx.lineTo(800, 800);
ctx.stroke();

// 域标签
ctx.fillStyle = '#e74c3c';
ctx.font = '11px Arial';
ctx.textAlign = 'center';
const mid5 = insS / 2;
const midIns = insS + (insE - insS) / 2;
const mid3 = insE + (L - insE) / 2;
ctx.fillText("5' tRNA", mid5/L*800, 15);
ctx.fillText("Insert", midIns/L*800, 15);
ctx.fillText("3' tRNA", mid3/L*800, 15);
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='单条 contact map 可视化 → HTML')
    parser.add_argument('--npy', type=str, required=True, help='.npy contact map 文件')
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    parser.add_argument('--out', type=str, default=None, help='输出 HTML 路径 (默认: 同目录同名 .html)')
    args = parser.parse_args()

    cmap = np.load(args.npy)
    name = os.path.splitext(os.path.basename(args.npy))[0]

    out = args.out or os.path.splitext(args.npy)[0] + '.html'
    html = cmap_to_html(cmap, title=name, trna_5end=args.trna_5end, trna_3end=args.trna_3end)

    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)

    total = int(cmap.sum() / 2)
    L = cmap.shape[0]
    cross = int(cmap[:args.trna_5end, L - args.trna_3end:].sum())
    print(f"[{name}] {L}nt  {total} pairs  cross={cross}")
    print(f"→ {out}")


if __name__ == '__main__':
    main()

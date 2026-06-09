#!/usr/bin/env python3
"""
Contact map (.npy) → RNA 二级结构图 (HTML arc diagram)

直接在接触图上画弧，不经过 dot-bracket 中转，零信息丢失。

用法:
  python draw_ss_from_npy.py \
      --npy ./chimera_predictions/bpRNA_CRW_15563.npy \
      --pkl ./data/chimeras.pkl
"""

import numpy as np
import argparse
import pickle
import os
import json


def find_sequence_in_pkl(pkl_path, npy_name):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    key = os.path.splitext(os.path.basename(npy_name))[0]
    for item in data:
        name = item.get('trna_name', '') if isinstance(item, dict) else getattr(item, 'trna_name', '')
        if key in name:
            seq = item['sequence'] if isinstance(item, dict) else item.sequence
            return seq, name
    return None, None


def build_html(seq, cmap, trna_5end=43, trna_3end=15, title=""):
    L = len(seq)
    ins_s = trna_5end
    ins_e = L - trna_3end

    # 收集所有配对
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            if cmap[i, j] > 0:
                pairs.append((i, j))

    # 分类
    cross_5_3 = [(i, j) for i, j in pairs if i < ins_s and j >= ins_e]
    t5_internal = [(i, j) for i, j in pairs if i < ins_s and j < ins_s]
    t3_internal = [(i, j) for i, j in pairs if i >= ins_e and j >= ins_e]
    ins_internal = [(i, j) for i, j in pairs if i >= ins_s and i < ins_e and j >= ins_s and j < ins_e]

    # HTML
    pairs_json = json.dumps([
        {"i": int(i), "j": int(j), "domain": (
            "cross" if (i < ins_s and j >= ins_e) else
            "t5" if j < ins_s else
            "t3" if i >= ins_e else
            "ins"
        )}
        for i, j in pairs
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>RNA SS - {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 10px; }}
.header {{ text-align: center; padding: 10px 0 5px; }}
.header h2 {{ font-size: 16px; color: #ccc; }}
.header .stats {{ font-size: 12px; color: #888; margin-top: 3px; }}
.legend {{ display: flex; justify-content: center; gap: 16px; margin: 8px 0; font-size: 11px; }}
.legend span {{ display: flex; align-items: center; gap: 4px; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.legend .dot.cross {{ background: #e67e22; }}
.legend .dot.t5 {{ background: #3498db; }}
.legend .dot.ins {{ background: #2ecc71; }}
.legend .dot.t3 {{ background: #9b59b6; }}
#canvas {{ display: block; margin: 0 auto; }}
.domain-bar {{ display: flex; justify-content: center; margin-top: 5px; font-size: 10px; font-family: monospace; gap: 2px; }}
.domain-bar .block {{ width: 3px; height: 14px; display: inline-block; border-radius: 1px; }}
.domain-bar .t {{ background: #3498db; }}
.domain-bar .i {{ background: #2ecc71; }}
</style>
</head>
<body>
<div class="header">
  <h2>{title} ({L} nt)</h2>
  <div class="stats">
    pairs={len(pairs)} &nbsp;|&nbsp;
    cross={len(cross_5_3)} &nbsp;|&nbsp;
    tRNA5={len(t5_internal)} &nbsp;|&nbsp;
    ins={len(ins_internal)} &nbsp;|&nbsp;
    tRNA3={len(t3_internal)}
  </div>
</div>
<div class="legend">
  <span><b class="dot cross"></b> 5'↔3' cross</span>
  <span><b class="dot t5"></b> 5' tRNA</span>
  <span><b class="dot ins"></b> Insert</span>
  <span><b class="dot t3"></b> 3' tRNA</span>
</div>
<canvas id="canvas"></canvas>
<div class="domain-bar" id="domainBar"></div>
<script>
const L = {L};
const insS = {ins_s};
const insE = {ins_e};
const pairs = {pairs_json};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ── 自适应布局 ──
const baseWidth = 6;
const arcHeight = 60;
const marginX = 5;
const marginTop = 20;
const marginBottom = 30;

// 紧凑布局：bases on bottom, arcs above
const totalW = L * baseWidth + marginX * 2;
// 找出最宽嵌套深度
const depthCount = new Array(L).fill(0);
const depthStack = [];
for (const p of pairs) {{
    depthStack.push(p);
}}
// 简化：高度随 cross-pair 数量自适应
const crossCount = pairs.filter(p => p.domain === 'cross').length;
const innerCount = pairs.filter(p => p.domain !== 'cross').length;
const arcTotalH = Math.max(crossCount, innerCount) * 3 + 80;
const height = arcTotalH + 80;

canvas.width = totalW;
canvas.height = height;
ctx.clearRect(0, 0, canvas.width, canvas.height);

// ── 域分段背景 ──
const bgY = height - 50;
const barH = 40;
ctx.fillStyle = 'rgba(52,152,219,0.15)';
ctx.fillRect(marginX, bgY, insS * baseWidth, barH);
ctx.fillStyle = 'rgba(46,204,113,0.15)';
ctx.fillRect(marginX + insS * baseWidth, bgY, (insE - insS) * baseWidth, barH);
ctx.fillStyle = 'rgba(155,89,182,0.15)';
ctx.fillRect(marginX + insE * baseWidth, bgY, (L - insE) * baseWidth, barH);

// ── 碱基标尺 ──
ctx.fillStyle = '#aaa';
ctx.font = '8px monospace';
ctx.textAlign = 'center';
const seq = "{seq}";
for (let i = 0; i < L; i++) {{
    const x = marginX + i * baseWidth + baseWidth / 2;
    ctx.fillText(seq[i], x, height - 10);
    // 小竖线
    if (i % 10 === 0) {{
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, height - 50);
        ctx.lineTo(x, height - 15);
        ctx.stroke();
    }}
}}

// 域分隔线
ctx.strokeStyle = '#e74c3c';
ctx.lineWidth = 1.5;
ctx.setLineDash([4, 3]);
[insS, insE].forEach(pos => {{
    const x = marginX + pos * baseWidth;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
}});
ctx.setLineDash([]);

// ── 分层画弧 ──
const domainColors = {{
    cross: '#e67e22',
    t5:    '#3498db',
    ins:   '#2ecc71',
    t3:    '#9b59b6',
}};

// 纵坐标分配：避免重叠
const levels = [];
for (const p of pairs) {{
    let level = 0;
    for (const q of levels) {{
        if (p.i < q.i && q.i < p.j && q.j < p.j) {{
            level = Math.max(level, q.level + 1);
        }}
        if (q.i < p.i && p.i < q.j && p.j < q.j) {{
            level = Math.max(level, q.level + 1);
        }}
    }}
    p.level = level;
    levels.push(p);
}}

for (const p of pairs) {{
    const x1 = marginX + p.i * baseWidth + baseWidth / 2;
    const x2 = marginX + p.j * baseWidth + baseWidth / 2;
    const midX = (x1 + x2) / 2;
    const span = x2 - x1;

    const baseY = height - 55;
    const thisArcH = Math.min(span * 0.35, 100);
    const levelOffset = p.level * 5;
    const cy = baseY - thisArcH / 2 - levelOffset;

    ctx.strokeStyle = domainColors[p.domain] || '#aaa';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x1, baseY);
    ctx.quadraticCurveTo(midX, cy, x2, baseY);
    ctx.stroke();

    // 端点小圆
    ctx.fillStyle = domainColors[p.domain] || '#aaa';
    ctx.beginPath();
    ctx.arc(x1, baseY, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x2, baseY, 2, 0, Math.PI * 2);
    ctx.fill();
}}

// ── 域标签 ──
ctx.fillStyle = '#fff';
ctx.font = '10px Arial';
ctx.textAlign = 'center';
const mid5 = insS / 2;
const midIns = insS + (insE - insS) / 2;
const mid3 = insE + (L - insE) / 2;
ctx.fillText("5' tRNA", marginX + mid5 * baseWidth, 12);
ctx.fillText("Insert",  marginX + midIns * baseWidth, 12);
ctx.fillText("3' tRNA", marginX + mid3 * baseWidth, 12);

</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, required=True, help='.npy contact map')
    parser.add_argument('--seq', type=str, default=None, help='序列')
    parser.add_argument('--pkl', type=str, default=None, help='chimeras.pkl')
    parser.add_argument('--idx', type=str, default=None)
    parser.add_argument('--trna_5end', type=int, default=43)
    parser.add_argument('--trna_3end', type=int, default=15)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    cmap = np.load(args.npy)
    L = cmap.shape[0]

    seq = args.seq
    name = None
    if seq is None and args.pkl:
        seq, name = find_sequence_in_pkl(args.pkl, args.idx or args.npy)
    if seq is None:
        print("[错误] 需要 --seq 或 --pkl")
        return
    seq = seq[:L].upper().replace('T', 'U')
    name = name or os.path.splitext(os.path.basename(args.npy))[0]

    html = build_html(seq, cmap, args.trna_5end, args.trna_3end, title=name)
    out = args.out or os.path.splitext(args.npy)[0] + '_ss.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"→ {out}")


if __name__ == '__main__':
    main()

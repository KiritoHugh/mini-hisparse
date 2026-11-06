"""Deterministic, explicitly laid out SVG figures for the standalone design.

Coordinates are intentional: each panel carries one argument, with no automatic
routing or crossed edges. The same SVG files are used by GitHub and the PDF.
"""

from html import escape
from math import hypot
from pathlib import Path

INK = "#20344a"
MUTED = "#607184"
LINE = "#96a4b3"
BLUE = ("#eaf0f8", "#7d9ab8")
GREEN = ("#e7f3ee", "#76a18f")
AMBER = ("#faf1e1", "#c8ab73")
GRAY = ("#f3f5f7", "#b2bdc8")


class Figure:
    def __init__(self, height, title, description):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" '
            f'viewBox="0 0 900 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="900" height="{height}" fill="white"/>',
            '<g font-family="Noto Sans CJK SC, sans-serif">',
        ]

    def rect(self, x, y, w, h, palette=GRAY, radius=7):
        fill, stroke = palette
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )

    def text(self, x, y, value, size=20, color=INK, anchor="middle", weight=400):
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{escape(value)}</text>'
        )

    def box(self, x, y, w, h, lines, palette=GRAY, size=20):
        self.rect(x, y, w, h, palette)
        spacing = 27
        first = y + h / 2 - (len(lines) - 1) * spacing / 2 + size * 0.35
        for i, line in enumerate(lines):
            self.text(x + w / 2, first + i * spacing, line, size=size)

    def arrow(self, points, dashed=False):
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{MUTED}" stroke-width="1.6"{dash}/>'
        )
        # Explicit geometry avoids SVG marker rotation differences in PDF engines.
        (x0, y0), (x, y) = points[-2:]
        length = hypot(x - x0, y - y0)
        dx, dy = (x - x0) / length, (y - y0) / length
        back_x, back_y = x - 8 * dx, y - 8 * dy
        vertices = [(x, y), (back_x - 4 * dy, back_y + 4 * dx), (back_x + 4 * dy, back_y - 4 * dx)]
        triangle = " ".join(f"{vx},{vy}" for vx, vy in vertices)
        self.parts.append(f'<polygon points="{triangle}" fill="{MUTED}"/>')

    def save(self, path):
        path.write_text("\n".join([*self.parts, "</g></svg>\n"]), encoding="utf-8")


def overview():
    f = Figure(250, "逻辑选择与两级存储", "上方为四步计算链；下方为独立 host 与共享 GPU 池。")
    for x, label in zip((15, 245, 475, 705), ("逻辑选择", "地址解析", "缺失行加载", "Attention")):
        f.box(x, 15, 180, 56, [label], BLUE if label == "Attention" else GRAY)
    for x in (195, 425, 655):
        f.arrow([(x, 43), (x + 50, 43)])
    f.text(450, 107, "只为本步所选数据解析地址与准备落点", size=19, color=MUTED)
    f.box(15, 139, 245, 92, ["Host backing", "保存完整逻辑历史"], AMBER)
    f.rect(280, 126, 605, 114, ("#ffffff", LINE))
    f.text(582, 151, "同一个 GPU KV tensor", size=19, weight=500)
    f.box(297, 165, 270, 59, ["Resident · 驻留完整页"], BLUE)
    f.box(587, 165, 280, 59, ["Hot · 缓存选中行"], GREEN)
    return f


def addressing():
    f = Figure(237, "三条地址解析路径", "优先 resident，随后 hot tag，最后才分配 miss 落点并加载。")
    rows = [
        (15, "Resident hit", "resident[p] = b", "b × P + offset", "直接可读", BLUE),
        (91, "Hot hit", "source[p] → host row", "tag → hot slot", "直接可读", GREEN),
        (167, "Miss", "source[p] → host row", "分配 slot + gather", "ready 后可读", AMBER),
    ]
    for y, label, first, second, last, palette in rows:
        f.text(12, y + 36, label, size=19, anchor="start", weight=500)
        f.box(170, y, 242, 57, [first], palette, size=19)
        f.arrow([(412, y + 28), (447, y + 28)])
        f.box(447, y, 245, 57, [second], palette, size=19)
        f.arrow([(692, y + 28), (727, y + 28)])
        f.box(727, y, 160, 57, [last], palette, size=19)
    return f


def lifecycle():
    f = Figure(
        278,
        "页回收与临时行就绪",
        "页经历 dirty、clean、reclaimable、host-only；hot 行单独经历分配、加载和就绪。",
    )
    f.text(12, 25, "逻辑页：是否仍拥有 resident 映射", size=20, anchor="start", weight=500)
    states = [
        (12, "Dirty", "GPU 原件受保护", BLUE),
        (250, "Clean", "host 镜像已完成", BLUE),
        (488, "Reclaimable", "仍可直接命中", BLUE),
        (726, "Host-only", "resident = −1", AMBER),
    ]
    for x, title, subtitle, palette in states:
        f.box(x, 63, 162, 70, [title, subtitle], palette, size=18)
    for x, label in ((174, "镜像完成"), (412, "资格满足"), (650, "实际复用")):
        f.text(x + 38, 54, label, size=17, color=MUTED)
        f.arrow([(x, 98), (x + 76, 98)])
    f.text(12, 181, "Hot 行：只改变副本，不改变逻辑页状态", size=20, anchor="start", weight=500)
    for x, label in ((12, "旧副本 / 空 slot"), (340, "新 tag · loading"), (668, "Ready · 可读")):
        f.box(x, 204, 220, 57, [label], GREEN, size=19)
    f.arrow([(232, 232), (340, 232)])
    f.text(286, 219, "分配", size=17, color=MUTED)
    f.arrow([(560, 232), (668, 232)])
    f.text(614, 219, "加载完成", size=17, color=MUTED)
    return f


def tails():
    f = Figure(
        251, "三块显存中的尾页轮转", "N=12 时保护 p1、p2；追加 t12 后保护 p2、p3，原 B1 安全复用。"
    )
    f.text(14, 55, "追加前", anchor="start", weight=500)
    f.text(14, 83, "N = 12", size=19, anchor="start", color=MUTED)
    f.text(14, 187, "追加 t12 后", anchor="start", weight=500)
    f.text(14, 215, "N = 13", size=19, anchor="start", color=MUTED)
    f.box(170, 26, 210, 80, ["B0 · Hot", "4 个可替换 rows"], GREEN)
    f.box(410, 26, 210, 80, ["B1 · Resident p1", "t4 … t7"], BLUE)
    f.box(650, 26, 235, 80, ["B2 · Resident p2", "t8 … t11"], BLUE)
    f.box(170, 158, 210, 80, ["B0 · Hot", "落点保持可用"], GREEN)
    f.box(410, 158, 210, 80, ["B1 · Resident p3", "t12 / 空 / 空 / 空"], BLUE)
    f.box(650, 158, 235, 80, ["B2 · Resident p2", "t8 … t11"], BLUE)
    f.arrow([(515, 106), (515, 158)])
    f.text(535, 139, "p1 已安全保存至 host", size=18, color=MUTED, anchor="start")
    return f


def step():
    f = Figure(
        256,
        "本步四个逻辑token的物理落点",
        "t0 hot hit，t5 miss 到 B0 offset1，t9 和 t12 resident hit；最终行号为0、1、9、4。",
    )
    entries = [
        (15, "t0", "Hot hit", "B0 : 0 → row 0", GREEN),
        (240, "t5", "Miss · 加载后", "B0 : 1 → row 1", AMBER),
        (465, "t9", "Resident hit", "B2 : 1 → row 9", BLUE),
        (690, "t12", "Resident hit", "B1 : 0 → row 4", BLUE),
    ]
    for x, token, kind, address, palette in entries:
        f.box(x, 10, 195, 63, [token, kind], palette, size=19)
        f.arrow([(x + 97.5, 73), (x + 97.5, 108)])
        f.box(x, 108, 195, 44, [address], palette, size=18)
    f.text(15, 192, "B0 内容", size=19, anchor="start", weight=500)
    for base, tokens in ((138, ("t0", "t1", "t2", "t3")), (550, ("t0", "t5", "t2", "t3"))):
        for j, token in enumerate(tokens):
            palette = AMBER if token == "t5" else GREEN
            f.box(base + j * 79, 170, 73, 44, [token], palette, size=19)
    f.arrow([(454, 192), (535, 192)])
    f.text(300, 245, "t0 本轮受保护", size=18, color=MUTED)
    f.text(705, 245, "只替换 t1；只加载 t5", size=18, color=MUTED)
    return f


def events():
    f = Figure(
        286,
        "基础解码的事件依赖",
        "compute 上追加、解析、attention；transfer 上镜像、gather，通过三个完成事件依赖连接。",
    )
    f.text(15, 28, "时间 / 依赖方向 →", size=18, color=MUTED, anchor="start")
    f.text(10, 85, "Compute", size=19, anchor="start", weight=500)
    f.text(10, 213, "Transfer", size=19, anchor="start", weight=500)
    f.box(140, 53, 150, 57, ["Append KV"], BLUE)
    f.box(370, 53, 190, 57, ["Select + resolve"], GRAY)
    f.box(740, 53, 145, 57, ["Attention"], BLUE)
    f.box(310, 181, 160, 57, ["D2H mirror"], AMBER)
    f.box(560, 181, 160, 57, ["Miss gather"], GREEN)
    f.arrow([(290, 81), (370, 81)])
    f.arrow([(260, 110), (260, 209), (310, 209)])
    f.text(248, 146, "append_ready", size=17, color=MUTED, anchor="end")
    f.arrow([(500, 110), (500, 146), (640, 146), (640, 181)])
    f.text(577, 135, "resolved", size=17, color=MUTED)
    f.arrow([(470, 209), (560, 209)])
    f.arrow([(720, 209), (812, 209), (812, 110)])
    f.text(802, 160, "hot_ready", size=17, color=MUTED, anchor="end")
    f.text(
        450, 276, "覆盖前另等待旧 reader / block fence；图示长度不代表耗时", size=18, color=MUTED
    )
    return f


def overlap():
    f = Figure(
        234,
        "可选的分区attention重叠",
        "解析后分为ready计算与miss加载，miss计算后合并softmax统计；整个分叉持有租约。",
    )
    f.box(15, 86, 145, 62, ["已完成解析", "并保护 hits"], GRAY, size=19)
    f.box(225, 16, 310, 61, ["Attention · ready 部分"], BLUE)
    f.box(225, 164, 195, 56, ["加载 misses"], GREEN)
    f.box(475, 164, 195, 56, ["Attention · miss"], BLUE, size=19)
    f.box(735, 86, 150, 62, ["合并统计量", "得到本步输出"], GRAY, size=19)
    f.arrow([(160, 117), (190, 117), (190, 46), (225, 46)])
    f.arrow([(190, 117), (190, 192), (225, 192)])
    f.arrow([(420, 192), (475, 192)])
    f.arrow([(535, 46), (810, 46), (810, 86)])
    f.arrow([(670, 192), (810, 192), (810, 148)])
    f.text(450, 122, "同一选择 · 两个不相交子集", size=19, color=MUTED)
    return f


def build_figures(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("overview", overview),
        ("addressing", addressing),
        ("lifecycle", lifecycle),
        ("tails", tails),
        ("step", step),
        ("events", events),
        ("overlap", overlap),
    ):
        draw().save(directory / f"{name}.svg")

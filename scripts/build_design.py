"""Build the standalone design PDF and SVG diagrams from versioned sources.

Install .[docs], Pango and Noto CJK fonts. Run from any directory:
    python scripts/build_design.py
"""

import argparse
import re
from html import escape
from pathlib import Path

from design_figures import build_figures


def main():
    from markdown_it import MarkdownIt
    from weasyprint import CSS, HTML

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "docs/partial-residency-design.pdf")
    args = parser.parse_args()
    source = root / "docs/partial-residency-design.md"
    assets = root / "docs/assets/design"
    build_figures(assets)
    markdown = source.read_text(encoding="utf-8")
    markdown = markdown.replace("<!-- pagebreak -->", '<div class="pagebreak"></div>')
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown)
    title = re.search(r"^# (.+)$", markdown, re.MULTILINE).group(1)
    html = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        '<meta name="description" content="部分驻留 KV Cache 的页面管理、共享池分配与异步解码设计">'
        f"</head><body>{body}</body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = HTML(string=html, base_url=str(source.parent)).render(
        stylesheets=[CSS(filename=str(assets / "print.css"))]
    )
    document.write_pdf(args.output)
    print(f"Wrote {len(document.pages)} pages: {args.output}")


if __name__ == "__main__":
    main()

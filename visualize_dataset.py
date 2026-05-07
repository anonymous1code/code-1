"""
Visualize / inspect the reviewer data sample.

Generates a self-contained HTML report with source and target image
thumbnails, metadata, and filtering controls.  For vcape-r rows where
the source image is stored as a physical_id string, the script downloads
the image on the fly via ImageDownloader (cached locally).

Usage:
    python scripts/visualize_datasample.py
    python scripts/visualize_datasample.py --dataset data/datasample_to_share --output report.html
    python scripts/visualize_datasample.py --max-thumb 512
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import load_from_disk
from PIL import Image as PILImage

from scripts.image_downloader import ImageDownloader

DATASET_PATH = "data/datasample_to_share"
OUTPUT_HTML = "data/datasample_to_share/report.html"
MAX_THUMB = 384  # max thumbnail dimension in pixels
SOURCE_IMG_CACHE = "data/imgs/pids"

_downloader = ImageDownloader()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_source_image(row: dict) -> PILImage.Image | None:
    """Resolve the source image from a dataset row.

    vcape-s rows have xsource_image as a PIL Image.
    vcape-r rows have physical_id (string) — downloaded via ImageDownloader.
    """
    src = row.get("xsource_image")
    if src is not None:
        return src.convert("RGB")

    pid = row.get("physical_id", "")
    if pid:
        try:
            path = _downloader.download_img(pid, SOURCE_IMG_CACHE)
            return PILImage.open(path).convert("RGB")
        except Exception as exc:
            print(f"  [WARN] Could not download source for physical_id={pid}: {exc}")
    return None


def _img_to_data_uri(img: PILImage.Image | None, max_dim: int) -> str:
    """Convert a PIL Image to a base64 data URI, or return a placeholder."""
    if img is None:
        return ""
    img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _esc(text: str | None) -> str:
    """HTML-escape a string."""
    if text is None:
        return ""
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f5; color: #222; padding: 24px; }
h1 { margin-bottom: 12px; }
.controls { margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.controls label { font-weight: 600; }
.controls select, .controls input { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(720px, 1fr)); gap: 16px; }
.card { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12);
        padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.card.hidden { display: none; }
.images { display: flex; gap: 12px; align-items: flex-start; }
.images img { max-height: 260px; border-radius: 4px; border: 1px solid #ddd; }
.placeholder { width: 200px; height: 200px; background: #eee; border-radius: 4px;
               display: flex; align-items: center; justify-content: center; color: #999; font-size: 13px; }
.meta { font-size: 13px; line-height: 1.6; }
.meta b { color: #555; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px;
         font-weight: 600; }
.badge-accepted { background: #d4edda; color: #155724; }
.badge-rejected { background: #f8d7da; color: #721c24; }
.badge-split { background: #d1ecf1; color: #0c5460; }
"""

_JS = """\
function applyFilters() {
  const split = document.getElementById('fSplit').value;
  const label = document.getElementById('fLabel').value;
  const reason = document.getElementById('fReason').value;
  document.querySelectorAll('.card').forEach(card => {
    const s = card.dataset.split;
    const l = card.dataset.label;
    const r = card.dataset.reason;
    const show = (split === 'all' || s === split)
              && (label === 'all' || l === label)
              && (reason === 'all' || r === reason);
    card.classList.toggle('hidden', !show);
  });
  // Update count
  const visible = document.querySelectorAll('.card:not(.hidden)').length;
  document.getElementById('count').textContent = visible;
}
"""


def build_html(rows: list[dict], max_thumb: int) -> str:
    """Build a self-contained HTML string from sample rows."""
    # Collect unique filter values
    splits = sorted({r["split"] for r in rows})
    labels = sorted({r["label_norm"] for r in rows})
    reasons = sorted({r["rejection_reason"] for r in rows if r["rejection_reason"]})

    def _options(values: list[str]) -> str:
        opts = '<option value="all">All</option>\n'
        for v in values:
            opts += f'<option value="{_esc(v)}">{_esc(v)}</option>\n'
        return opts

    cards_html = []
    for i, r in enumerate(rows):
        src_uri = r["_src_uri"]
        tgt_uri = r["_tgt_uri"]

        src_tag = (
            f'<img src="{src_uri}" alt="source">'
            if src_uri
            else '<div class="placeholder">source N/A</div>'
        )
        tgt_tag = (
            f'<img src="{tgt_uri}" alt="target">'
            if tgt_uri
            else '<div class="placeholder">target N/A</div>'
        )

        label_cls = "badge-accepted" if r["label_norm"] == "accepted" else "badge-rejected"

        card = f"""\
<div class="card" data-split="{_esc(r['split'])}" data-label="{_esc(r['label_norm'])}" data-reason="{_esc(r['rejection_reason'])}">
  <div style="display:flex;gap:8px;align-items:center;">
    <span class="badge badge-split">{_esc(r['split'])}</span>
    <span class="badge {label_cls}">{_esc(r['label_norm'])}</span>
    <span style="font-size:12px;color:#888;">#{i}</span>
  </div>
  <div class="images">
    <div><div style="font-size:11px;color:#888;margin-bottom:4px;">Source</div>{src_tag}</div>
    <div><div style="font-size:11px;color:#888;margin-bottom:4px;">Target</div>{tgt_tag}</div>
  </div>
  <div class="meta">
    <b>Product type:</b> {_esc(r['product_type'])} &nbsp;|&nbsp;
    <b>Pose:</b> {_esc(r['pose'])} &nbsp;|&nbsp;
    <b>Rejection reason:</b> {_esc(r['rejection_reason']) or '—'}<br>
    <b>Prompt:</b> {_esc(r['input_prompt'])}<br>
    <b>Description:</b> {_esc(r['object_description'])}
  </div>
</div>"""
        cards_html.append(card)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>V-CAPE Data Sample — Reviewer Inspection</title>
<style>{_CSS}</style>
</head>
<body>
<h1>V-CAPE Data Sample</h1>
<p style="margin-bottom:12px;color:#555;">
  {len(rows)} samples from vcape-r and vcape-s for reviewer quality check.
  Showing <span id="count">{len(rows)}</span> samples.
</p>
<div class="controls">
  <label>Split:</label>
  <select id="fSplit" onchange="applyFilters()">{_options(splits)}</select>
  <label>Label:</label>
  <select id="fLabel" onchange="applyFilters()">{_options(labels)}</select>
  <label>Rejection reason:</label>
  <select id="fReason" onchange="applyFilters()">{_options(reasons)}</select>
</div>
<div class="grid">
{"".join(cards_html)}
</div>
<script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reviewer data sample")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_HTML)
    parser.add_argument("--max-thumb", type=int, default=MAX_THUMB,
                        help="Max thumbnail dimension in pixels")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    ds = load_from_disk(args.dataset)
    print(f"  {len(ds)} rows")

    rows: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]

        # Normalise label
        raw_label = (row.get("label") or "").strip().lower()
        label_norm = "accepted" if raw_label in ("accepted", "accept") else "rejected"

        # Resolve images
        src_img = _get_source_image(row)
        tgt_img = row.get("xtarget_image")
        if tgt_img is not None:
            tgt_img = tgt_img.convert("RGB")

        rows.append(
            {
                "split": row.get("split", "unknown"),
                "label_norm": label_norm,
                "rejection_reason": row.get("rejection_reason", "") or "",
                "product_type": row.get("product_type", ""),
                "pose": row.get("pose", ""),
                "input_prompt": row.get("input_prompt", ""),
                "object_description": row.get("object_description", ""),
                "_src_uri": _img_to_data_uri(src_img, args.max_thumb),
                "_tgt_uri": _img_to_data_uri(tgt_img, args.max_thumb),
            }
        )
        print(f"  [{i+1}/{len(ds)}] {rows[-1]['split']} / {rows[-1]['label_norm']} / {rows[-1]['rejection_reason']}")

    html_content = build_html(rows, args.max_thumb)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html_content)
    print(f"\nReport saved to {args.output}")
    print(f"Open in a browser to inspect the samples.")


if __name__ == "__main__":
    main()

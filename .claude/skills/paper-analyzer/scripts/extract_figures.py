#!/usr/bin/env python3
"""논문 PDF에서 Figure/Table을 추출하는 스크립트.

권장 모드는 ``auto``: 캡션 패턴(Figure N: / Table N:)을 정규식으로 찾아서
그 위쪽의 그래픽/텍스트 영역을 자동으로 묶어 캡션과 함께 한 박스로 잘라낸다.
좌표를 직접 추정하는 ``crop`` 모드는 자동 검출이 실패한 figure에만 사용한다.

사용법:
  # 기본: 자동 검출 + 추출 (가장 권장)
  python extract_figures.py <pdf> <out_dir>                    # mode=auto 기본
  python extract_figures.py <pdf> <out_dir> --select "Figure 1,Table 1"

  # 검출 결과만 보기 (이미지 저장 안 함)
  python extract_figures.py <pdf> <out_dir> --mode detect

  # 자동 검출이 실패한 경우의 fallback
  python extract_figures.py <pdf> <out_dir> --mode crop \\
    --regions '[{"page": 2, "bbox": [50, 100, 500, 400]}]'

  # 그 외 도구
  python extract_figures.py <pdf> <out_dir> --mode info
  python extract_figures.py <pdf> <out_dir> --mode scan
  python extract_figures.py <pdf> <out_dir> --mode page --pages 2,5
"""

import argparse
import json
import os
import re
import sys

import fitz  # PyMuPDF


CAPTION_PATTERN = re.compile(
    r"^(?i:(Figure|Fig\.|Table))\s+(\d+)(?:\s*[:.]|\s+[A-Z])",
)


# ============================================================================
# Helpers — text/graphics extraction
# ============================================================================

def _block_text(block: dict) -> str:
    return "".join(
        span["text"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ).strip()


def get_text_and_caption_blocks(page) -> tuple[list[dict], list[dict]]:
    """페이지의 텍스트 블록 전체 + 그 중 caption인 것들을 반환."""
    text_blocks: list[dict] = []
    caption_blocks: list[dict] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type", 1) != 0:
            continue
        text = _block_text(block)
        if not text:
            continue
        info = {"bbox": tuple(block["bbox"]), "text": text}
        text_blocks.append(info)
        match = CAPTION_PATTERN.match(text)
        if match:
            caption_blocks.append({
                **info,
                "type": match.group(1).rstrip("."),
                "number": int(match.group(2)),
            })
    return text_blocks, caption_blocks


def get_graphical_regions(page) -> list[tuple[str, tuple]]:
    """이 페이지의 모든 그래픽 영역(image bbox + drawing bbox)."""
    regions: list[tuple[str, tuple]] = []

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for r in rects:
            if r.width > 5 and r.height > 5:
                regions.append(("image", (r.x0, r.y0, r.x1, r.y1)))

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        if rect.width > 8 and rect.height > 8:
            regions.append(("drawing", (rect.x0, rect.y0, rect.x1, rect.y1)))

    return regions


# ============================================================================
# Layout detection
# ============================================================================

def detect_column_layout(page) -> tuple[str, float]:
    """페이지가 1-column인지 2-column인지 추정."""
    text_blocks, _ = get_text_and_caption_blocks(page)
    page_w = page.rect.width
    mid = page_w / 2

    if not text_blocks:
        return ("single", mid)

    left = sum(1 for b in text_blocks if b["bbox"][2] < mid + 5)
    right = sum(1 for b in text_blocks if b["bbox"][0] > mid - 5)
    full = sum(1 for b in text_blocks
               if b["bbox"][0] < mid - 10 and b["bbox"][2] > mid + 10)

    if left >= 3 and right >= 3 and full < min(left, right):
        return ("double", mid)
    return ("single", mid)


def determine_caption_column(caption_bbox, page_rect, layout, mid_x) -> str:
    """캡션이 left / right / full 중 어느 컬럼에 속하는지."""
    cx0, _, cx1, _ = caption_bbox
    cw = cx1 - cx0
    if layout == "single" or cw > page_rect.width * 0.6:
        return "full"
    if cx1 <= mid_x + 15:
        return "left"
    if cx0 >= mid_x - 15:
        return "right"
    return "full"


def _column_contains(col: str, mid_x: float, bbox) -> bool:
    bx0, _, bx1, _ = bbox
    if col == "full":
        return True
    if col == "left":
        return bx1 <= mid_x + 30
    if col == "right":
        return bx0 >= mid_x - 30
    return True


# ============================================================================
# Bbox merging
# ============================================================================

def merge_overlapping(bboxes: list[tuple], gap_tol: float = 18) -> list[tuple]:
    """근접/겹치는 bbox들을 하나로 병합."""
    if not bboxes:
        return []
    boxes = [list(b) for b in bboxes]
    changed = True
    while changed:
        changed = False
        used = [False] * len(boxes)
        merged: list[list[float]] = []
        for i, cur in enumerate(boxes):
            if used[i]:
                continue
            cur = list(cur)
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                ox0, oy0, ox1, oy1 = boxes[j]
                cx0, cy0, cx1, cy1 = cur
                near = (cx1 + gap_tol >= ox0 and ox1 + gap_tol >= cx0
                        and cy1 + gap_tol >= oy0 and oy1 + gap_tol >= cy0)
                if near:
                    cur = [min(cx0, ox0), min(cy0, oy0),
                           max(cx1, ox1), max(cy1, oy1)]
                    used[j] = True
                    changed = True
            merged.append(cur)
        boxes = merged
    return [tuple(b) for b in boxes]


# ============================================================================
# Figure bbox detection
# ============================================================================

def detect_figure_bbox(
    page,
    caption: dict,
    layout_info: tuple[str, float],
    *,
    max_gap: float = 50,
    padding: float = 5,
) -> list[float]:
    """캡션을 기준으로 figure 영역(캡션 포함)을 자동 추정.

    Figure 캡션은 보통 그림 아래, Table 캡션은 보통 표 위에 있다는
    관습을 따라 검색 방향을 결정한다.
    """
    page_rect = page.rect
    layout, mid_x = layout_info
    caption_bbox = caption["bbox"]
    cap_y0, cap_y1 = caption_bbox[1], caption_bbox[3]
    is_table = caption["type"].lower().startswith("tab")
    search_above = not is_table  # Figure: 위 / Table: 아래

    col = determine_caption_column(caption_bbox, page_rect, layout, mid_x)
    text_blocks, _ = get_text_and_caption_blocks(page)
    all_graphics = [g for _, g in get_graphical_regions(page)]

    def _passes_position(bbox) -> bool:
        if search_above:
            return bbox[3] <= cap_y0 + 2
        return bbox[1] >= cap_y1 - 2

    def _is_text_dominated(graphic_bbox: tuple, threshold: float = 0.5) -> bool:
        """그래픽 영역의 절반 이상이 텍스트로 채워져 있으면 본문 박스로 판단.

        Abstract 박스, Definition 박스 등을 figure 후보에서 제외하기 위함.
        """
        gx0, gy0, gx1, gy1 = graphic_bbox
        g_area = (gx1 - gx0) * (gy1 - gy0)
        if g_area < 2000:  # 너무 작은 박스는 라벨일 수도
            return False
        text_area = 0.0
        for tb in text_blocks:
            tx0, ty0, tx1, ty1 = tb["bbox"]
            ix0, iy0 = max(tx0, gx0), max(ty0, gy0)
            ix1, iy1 = min(tx1, gx1), min(ty1, gy1)
            if ix1 > ix0 and iy1 > iy0:
                text_area += (ix1 - ix0) * (iy1 - iy0)
        return (text_area / g_area) > threshold

    def _is_inside_any(bbox: tuple, others: list[tuple], slack: float = 3) -> bool:
        bx0, by0, bx1, by1 = bbox
        for ox0, oy0, ox1, oy1 in others:
            if (bx0 >= ox0 - slack and by0 >= oy0 - slack
                    and bx1 <= ox1 + slack and by1 <= oy1 + slack):
                return True
        return False

    def _has_paragraph_between(graphic_bbox: tuple) -> bool:
        """캡션과 graphic 사이에 본문 텍스트 단락이 끼어있는가?

        그래픽 내부의 라벨(figure 안의 텍스트)은 paragraph가 아니므로 무시한다.
        본문 단락의 기준: 높이 ≥ 12pt + 폭 ≥ 80pt + 다른 그래픽 영역 안에 있지 않음.
        """
        if search_above:
            gap_low, gap_high = graphic_bbox[3], cap_y0
        else:
            gap_low, gap_high = cap_y1, graphic_bbox[1]
        if gap_high - gap_low < 12:
            return False
        other_graphics = [g for g in all_graphics if g != graphic_bbox]
        for tb in text_blocks:
            tb_bbox = tb["bbox"]
            if tb_bbox == caption_bbox:
                continue
            if not _column_contains(col, mid_x, tb_bbox):
                continue
            tby0, tby1 = tb_bbox[1], tb_bbox[3]
            if tby0 < gap_low - 2 or tby1 > gap_high + 2:
                continue
            if (tby1 - tby0) < 12:
                continue
            if (tb_bbox[2] - tb_bbox[0]) < 80:
                continue
            # 다른 그래픽(figure box) 내부 라벨은 무시
            if _is_inside_any(tb_bbox, other_graphics):
                continue
            return True
        return False

    # 1) 그래픽 영역 후보
    candidates: list[tuple] = []
    for bbox in all_graphics:
        if not _passes_position(bbox):
            continue
        if not _column_contains(col, mid_x, bbox):
            continue
        if _is_text_dominated(bbox):
            continue  # abstract/definition 박스
        if _has_paragraph_between(bbox):
            continue  # 다른 그림/박스로 판단
        candidates.append(bbox)

    # 2) Table은 텍스트 블록도 후보 (표는 그리기가 아닌 텍스트 격자)
    if is_table:
        for tb in text_blocks:
            bbox = tb["bbox"]
            text = tb.get("text", "")
            if bbox == caption_bbox:
                continue
            if not _passes_position(bbox):
                continue
            if not _column_contains(col, mid_x, bbox):
                continue
            dist = bbox[1] - cap_y1 if not search_above else cap_y0 - bbox[3]
            if dist > max_gap * 12:  # 표는 길 수 있어 거리 제한 완화
                continue
            # 본문 단락(폭 넓고 길이 긴 산문)은 표 후보가 아님
            width = bbox[2] - bbox[0]
            if width > 250 and len(text) > 80:
                continue
            candidates.append(bbox)

    # 3) 인접 후보 병합
    merged = merge_overlapping(candidates, gap_tol=22)

    # 4) 캡션 근처에서 시작해 연결된 영역만 채택
    relevant: list[tuple] = []
    if search_above:
        merged.sort(key=lambda b: b[3], reverse=True)  # y1 큰 순
        cur_edge = cap_y0
        for b in merged:
            if b[3] >= cur_edge - max_gap:
                relevant.append(b)
                cur_edge = min(cur_edge, b[1])
            else:
                break
    else:
        # Table: 첫 블록은 캡션 직후, 후속은 표 행 사이 갭(≤12pt)만 허용
        # 본문 단락이 표 끝에 인접해도 갭이 그보다 크므로 자연스럽게 차단됨
        merged.sort(key=lambda b: b[1])
        cur_edge = cap_y1
        first = True
        for b in merged:
            gap = b[1] - cur_edge
            if first:
                if gap <= max_gap:
                    relevant.append(b)
                    cur_edge = b[3]
                    first = False
                else:
                    break
            else:
                if gap <= 12:
                    relevant.append(b)
                    cur_edge = b[3]
                else:
                    break

    # 5) 그래도 비었으면 가장 가까운 영역
    if not relevant and merged:
        relevant = [merged[0]]

    # 6) 최종 합집합 (figure + 캡션)
    if relevant:
        all_boxes = relevant + [caption_bbox]
        x0 = min(b[0] for b in all_boxes)
        y0 = min(b[1] for b in all_boxes)
        x1 = max(b[2] for b in all_boxes)
        y1 = max(b[3] for b in all_boxes)
    else:
        x0, y0, x1, y1 = caption_bbox

    # 7) 컬럼에 맞춰 x 정규화 (단일 컬럼 figure를 컬럼 폭으로 깔끔하게)
    page_w = page_rect.width
    body_left = page_w * 0.06
    body_right = page_w * 0.94
    if col == "full":
        x0 = max(x0, body_left)
        x1 = min(x1, body_right)
    elif col == "left":
        x0 = max(x0, body_left)
        x1 = min(max(x1, mid_x - 3), mid_x + 3)
    elif col == "right":
        x0 = max(min(x0, mid_x + 3), mid_x - 3)
        x1 = min(x1, body_right)

    # 8) 패딩
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(page_w, x1 + padding)
    y1 = min(page_rect.height, y1 + padding)

    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def detect_figures(pdf_path: str, types: list[str] | None = None) -> list[dict]:
    """PDF 전체에서 Figure/Table을 자동 검출."""
    type_filter = None
    if types:
        type_filter = {t.lower().rstrip(".") for t in types}

    doc = fitz.open(pdf_path)
    results: list[dict] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        layout_info = detect_column_layout(page)
        _, captions = get_text_and_caption_blocks(page)

        for caption in captions:
            ctype = caption["type"].lower().rstrip(".")
            if ctype.startswith("fig"):
                ctype = "figure"
            if type_filter and ctype not in type_filter:
                continue

            bbox = detect_figure_bbox(page, caption, layout_info)
            label_type = "Figure" if ctype == "figure" else "Table"
            results.append({
                "page": page_num + 1,
                "type": label_type,
                "number": caption["number"],
                "label": f"{label_type} {caption['number']}",
                "bbox": bbox,
                "caption": caption["text"][:200],
            })

    doc.close()
    return results


# ============================================================================
# Extraction modes
# ============================================================================

def auto_extract(
    pdf_path: str,
    output_dir: str,
    *,
    dpi: int = 200,
    types: list[str] | None = None,
    select: list[str] | None = None,
    start_index: int = 1,
) -> list[dict]:
    """자동 검출 → 추출."""
    figures = detect_figures(pdf_path, types=types)

    if select:
        wanted = {s.strip().lower().replace(" ", "") for s in select}
        figures = [f for f in figures
                   if f["label"].lower().replace(" ", "") in wanted]

    doc = fitz.open(pdf_path)
    results: list[dict] = []

    for idx, fig in enumerate(figures, start=start_index):
        page = doc[fig["page"] - 1]
        bbox = fitz.Rect(fig["bbox"])
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
        filename = f"image_{idx}.png"
        pix.save(os.path.join(output_dir, filename))
        results.append({
            "index": idx,
            "page": fig["page"],
            "label": fig["label"],
            "bbox": fig["bbox"],
            "filename": filename,
            "width": pix.width,
            "height": pix.height,
            "caption": fig["caption"],
        })

    doc.close()
    return results


def crop_regions(
    pdf_path: str,
    output_dir: str,
    regions: list[dict],
    *,
    dpi: int = 200,
    start_index: int = 1,
) -> list[dict]:
    """수동 bbox 영역 캡쳐. 자동 검출이 실패한 경우의 fallback."""
    doc = fitz.open(pdf_path)
    results: list[dict] = []

    for idx, region in enumerate(regions, start=start_index):
        page_num = region["page"] - 1
        if page_num < 0 or page_num >= len(doc):
            print(f"Warning: page {region['page']} out of range, skipping",
                  file=sys.stderr)
            continue
        page = doc[page_num]
        bbox = fitz.Rect(region["bbox"])
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox)
        filename = f"image_{idx}.png"
        pix.save(os.path.join(output_dir, filename))
        results.append({
            "index": idx,
            "page": region["page"],
            "filename": filename,
            "width": pix.width,
            "height": pix.height,
            "bbox": region["bbox"],
        })

    doc.close()
    return results


def scan_images(pdf_path: str, output_dir: str, min_size: int = 100) -> list[dict]:
    """PDF 내 임베디드 raster 이미지를 모두 추출 (탐색용)."""
    doc = fitz.open(pdf_path)
    results: list[dict] = []
    img_count = 0

    for page_num in range(len(doc)):
        for img_info in doc[page_num].get_images(full=True):
            xref = img_info[0]
            base = doc.extract_image(xref)
            if not base:
                continue
            w, h = base["width"], base["height"]
            if w < min_size or h < min_size:
                continue
            img_count += 1
            ext = base["ext"]
            data = base["image"]
            filename = f"scan_{img_count}_p{page_num + 1}.{ext}"
            with open(os.path.join(output_dir, filename), "wb") as f:
                f.write(data)
            results.append({
                "index": img_count, "page": page_num + 1, "filename": filename,
                "width": w, "height": h, "size_bytes": len(data),
            })

    doc.close()
    return results


def render_pages(pdf_path: str, output_dir: str, pages: list[int],
                 dpi: int = 200) -> list[dict]:
    """특정 페이지를 고해상도로 통째 렌더링."""
    doc = fitz.open(pdf_path)
    results: list[dict] = []
    for idx, page_num in enumerate(pages, 1):
        if page_num - 1 < 0 or page_num - 1 >= len(doc):
            print(f"Warning: page {page_num} out of range, skipping",
                  file=sys.stderr)
            continue
        page = doc[page_num - 1]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        filename = f"page_{page_num}.png"
        pix.save(os.path.join(output_dir, filename))
        results.append({
            "index": idx, "page": page_num, "filename": filename,
            "width": pix.width, "height": pix.height,
        })
    doc.close()
    return results


def get_pdf_info(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    info = {
        "total_pages": len(doc),
        "metadata": doc.metadata,
        "page_sizes": [],
    }
    for i in range(min(3, len(doc))):
        rect = doc[i].rect
        info["page_sizes"].append({
            "page": i + 1,
            "width": round(rect.width, 1),
            "height": round(rect.height, 1),
        })
    doc.close()
    return info


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="논문 PDF에서 Figure 추출")
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument("output_dir", help="이미지 저장 디렉토리")
    parser.add_argument(
        "--mode",
        choices=["auto", "detect", "crop", "scan", "page", "info"],
        default="auto",
        help="기본 auto. detect=검출만 / crop=수동 bbox / scan=임베디드 이미지 / page=페이지 렌더 / info=PDF 정보",
    )
    parser.add_argument("--regions", type=str, default="[]",
                        help='crop 모드용 JSON: [{"page":1,"bbox":[x0,y0,x1,y1]}, ...]')
    parser.add_argument("--pages", type=str, default="",
                        help="page 모드용 페이지 번호 (콤마 구분)")
    parser.add_argument("--dpi", type=int, default=200, help="렌더링 해상도 (기본 200)")
    parser.add_argument("--min-size", type=int, default=100,
                        help="scan 모드 최소 이미지 픽셀 크기")
    parser.add_argument("--types", type=str, default="Figure,Table",
                        help="auto/detect 모드의 캡션 타입 필터 (콤마 구분)")
    parser.add_argument("--select", type=str, default="",
                        help='auto 모드에서 추출할 라벨 (예: "Figure 1,Table 2")')
    parser.add_argument("--start-index", type=int, default=1,
                        help="출력 파일명 image_N.png의 시작 N (기본 1)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    select = ([s.strip() for s in args.select.split(",") if s.strip()]
              if args.select else None)

    if args.mode == "info":
        print(json.dumps(get_pdf_info(args.pdf_path), indent=2, ensure_ascii=False))
    elif args.mode == "scan":
        results = scan_images(args.pdf_path, args.output_dir, min_size=args.min_size)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{len(results)}개 이미지 추출 → {args.output_dir}", file=sys.stderr)
    elif args.mode == "page":
        pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
        results = render_pages(args.pdf_path, args.output_dir, pages, dpi=args.dpi)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{len(results)}개 페이지 렌더링 → {args.output_dir}", file=sys.stderr)
    elif args.mode == "crop":
        regions = json.loads(args.regions)
        results = crop_regions(args.pdf_path, args.output_dir, regions,
                               dpi=args.dpi, start_index=args.start_index)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{len(results)}개 영역 캡쳐 → {args.output_dir}", file=sys.stderr)
    elif args.mode == "detect":
        results = detect_figures(args.pdf_path, types=types)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{len(results)}개 figure/table 검출", file=sys.stderr)
    elif args.mode == "auto":
        results = auto_extract(args.pdf_path, args.output_dir,
                               dpi=args.dpi, types=types, select=select,
                               start_index=args.start_index)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{len(results)}개 figure/table 자동 추출 → {args.output_dir}",
              file=sys.stderr)


if __name__ == "__main__":
    main()

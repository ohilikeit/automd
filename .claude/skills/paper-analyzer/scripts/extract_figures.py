#!/usr/bin/env python3
"""
논문 PDF에서 Figure를 추출하는 스크립트.

사용법:
  # 모드 1: PDF 내 모든 임베디드 이미지 추출 (탐색용)
  python extract_figures.py <pdf_path> <output_dir> --mode scan

  # 모드 2: 특정 페이지의 특정 영역을 캡쳐 (정밀 추출)
  python extract_figures.py <pdf_path> <output_dir> --mode crop \
    --regions '[{"page": 2, "bbox": [50, 100, 500, 400]}, ...]'

  # 모드 3: 특정 페이지 전체를 고해상도로 렌더링
  python extract_figures.py <pdf_path> <output_dir> --mode page \
    --pages 2,5,8
"""

import argparse
import json
import os
import sys

import fitz  # PyMuPDF


def scan_images(pdf_path: str, output_dir: str, min_size: int = 100) -> list[dict]:
    """PDF 내 모든 임베디드 이미지를 추출한다. 작은 아이콘/로고는 필터링."""
    doc = fitz.open(pdf_path)
    results = []
    img_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            if not base_image:
                continue

            width = base_image["width"]
            height = base_image["height"]

            # 너무 작은 이미지 필터링 (아이콘, 로고 등)
            if width < min_size or height < min_size:
                continue

            img_count += 1
            ext = base_image["ext"]
            img_data = base_image["image"]

            filename = f"scan_{img_count}_p{page_num + 1}.{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(img_data)

            results.append({
                "index": img_count,
                "page": page_num + 1,
                "filename": filename,
                "width": width,
                "height": height,
                "size_bytes": len(img_data),
            })

    doc.close()
    return results


def crop_regions(pdf_path: str, output_dir: str, regions: list[dict], dpi: int = 200) -> list[dict]:
    """PDF의 특정 페이지에서 지정된 bbox 영역을 고해상도로 캡쳐한다."""
    doc = fitz.open(pdf_path)
    results = []

    for idx, region in enumerate(regions, 1):
        page_num = region["page"] - 1  # 0-indexed
        if page_num < 0 or page_num >= len(doc):
            print(f"Warning: page {region['page']} out of range, skipping", file=sys.stderr)
            continue

        page = doc[page_num]
        bbox = fitz.Rect(region["bbox"])

        # 고해상도 렌더링
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=bbox)

        filename = f"image_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        pix.save(filepath)

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


def render_pages(pdf_path: str, output_dir: str, pages: list[int], dpi: int = 200) -> list[dict]:
    """특정 페이지 전체를 고해상도 이미지로 렌더링한다."""
    doc = fitz.open(pdf_path)
    results = []

    for idx, page_num in enumerate(pages, 1):
        if page_num - 1 < 0 or page_num - 1 >= len(doc):
            print(f"Warning: page {page_num} out of range, skipping", file=sys.stderr)
            continue

        page = doc[page_num - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        filename = f"page_{page_num}.png"
        filepath = os.path.join(output_dir, filename)
        pix.save(filepath)

        results.append({
            "index": idx,
            "page": page_num,
            "filename": filename,
            "width": pix.width,
            "height": pix.height,
        })

    doc.close()
    return results


def get_pdf_info(pdf_path: str) -> dict:
    """PDF 기본 정보를 반환한다."""
    doc = fitz.open(pdf_path)
    info = {
        "total_pages": len(doc),
        "metadata": doc.metadata,
        "page_sizes": [],
    }
    for i in range(min(3, len(doc))):
        page = doc[i]
        rect = page.rect
        info["page_sizes"].append({
            "page": i + 1,
            "width": round(rect.width, 1),
            "height": round(rect.height, 1),
        })
    doc.close()
    return info


def main():
    parser = argparse.ArgumentParser(description="논문 PDF에서 Figure 추출")
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument("output_dir", help="이미지 저장 디렉토리")
    parser.add_argument("--mode", choices=["scan", "crop", "page", "info"], default="scan",
                        help="추출 모드: scan(전체스캔), crop(영역캡쳐), page(페이지렌더), info(PDF정보)")
    parser.add_argument("--regions", type=str, default="[]",
                        help='crop 모드용 JSON: [{"page": 1, "bbox": [x0, y0, x1, y1]}, ...]')
    parser.add_argument("--pages", type=str, default="",
                        help="page 모드용 페이지 번호 (콤마 구분): 2,5,8")
    parser.add_argument("--dpi", type=int, default=200, help="렌더링 해상도 (기본: 200)")
    parser.add_argument("--min-size", type=int, default=100, help="scan 모드 최소 이미지 크기 (기본: 100px)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "info":
        info = get_pdf_info(args.pdf_path)
        print(json.dumps(info, indent=2, ensure_ascii=False))

    elif args.mode == "scan":
        results = scan_images(args.pdf_path, args.output_dir, min_size=args.min_size)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n총 {len(results)}개 이미지 추출 완료 → {args.output_dir}", file=sys.stderr)

    elif args.mode == "crop":
        regions = json.loads(args.regions)
        results = crop_regions(args.pdf_path, args.output_dir, regions, dpi=args.dpi)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n총 {len(results)}개 영역 캡쳐 완료 → {args.output_dir}", file=sys.stderr)

    elif args.mode == "page":
        pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
        results = render_pages(args.pdf_path, args.output_dir, pages, dpi=args.dpi)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n총 {len(results)}개 페이지 렌더링 완료 → {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

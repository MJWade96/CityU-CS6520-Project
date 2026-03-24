"""
StatPearls download and processing helpers.

The extraction logic is centralized here so the CLI wrapper, corpus combiner,
and future pipelines can reuse the same implementation without duplication.
"""

from __future__ import annotations

import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm


STATPEARLS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
)
STATPEARLS_ARCHIVE = "statpearls_NBK430685.tar.gz"
EXTRACTED_DIR_NAME = "statpearls_NBK430685"
EXPECTED_NXML_COUNT = 3200


def ends_with_ending_punctuation(text: str) -> bool:
    return any(text.endswith(char) for char in (".", "?", "!"))


def concat_title_and_content(title: str, content: str) -> str:
    separator = " " if ends_with_ending_punctuation(title.strip()) else ". "
    return f"{title.strip()}{separator}{content.strip()}"


def extract_text(element: ET.Element) -> str:
    text = (element.text or "").strip()
    for child in element:
        text += (" " if text else "") + extract_text(child)
        if child.tail and child.tail.strip():
            text += (" " if text else "") + child.tail.strip()
    return text.strip()


def is_subtitle(element: ET.Element) -> bool:
    if element.tag != "p":
        return False
    if len(list(element)) != 1 or list(element)[0].tag != "bold":
        return False
    return not (list(element)[0].tail and list(element)[0].tail.strip())


def extract_statpearls_article(file_path: Path) -> List[Dict]:
    """
    Convert one NXML article into paragraph-level chunks.

    This is intentionally shared with the download pipeline so we only maintain
    one chunking implementation for StatPearls.
    """
    article_id = file_path.stem
    tree = ET.parse(file_path)
    root = tree.getroot()

    title_elem = root.find(".//title")
    title = title_elem.text if title_elem is not None and title_elem.text else article_id

    chunks: List[Dict] = []
    chunk_index = 0

    for section in root.findall(".//sec"):
        sec_title_elem = section.find("./title")
        sec_title = (
            sec_title_elem.text.strip()
            if sec_title_elem is not None and sec_title_elem.text
            else ""
        )
        prefix = " -- ".join(part for part in [title, sec_title] if part)
        last_chunk: Dict | None = None
        last_node = None

        for child in section:
            if is_subtitle(child):
                sub_title = extract_text(child)
                prefix = " -- ".join(part for part in [title, sec_title, sub_title] if part)
                last_chunk = None
                last_node = child
                continue

            if child.tag == "p":
                current_text = extract_text(child)
                if (
                    last_chunk
                    and len(current_text) < 200
                    and len(last_chunk["content"] + current_text) < 1000
                ):
                    last_chunk["content"] = f"{last_chunk['content']} {current_text}".strip()
                    last_chunk["contents"] = concat_title_and_content(
                        last_chunk["title"], last_chunk["content"]
                    )
                else:
                    last_chunk = {
                        "id": f"{article_id}_{chunk_index}",
                        "title": prefix or title,
                        "content": current_text,
                        "contents": concat_title_and_content(prefix or title, current_text),
                        "source": "statpearls",
                    }
                    chunks.append(last_chunk)
                    chunk_index += 1
            elif child.tag == "list":
                list_text = [extract_text(item) for item in child]
                joined = " ".join(list_text)
                if last_chunk and len(last_chunk["content"] + joined) < 1000:
                    last_chunk["content"] = f"{last_chunk['content']} {joined}".strip()
                    last_chunk["contents"] = concat_title_and_content(
                        last_chunk["title"], last_chunk["content"]
                    )
                elif len(joined) < 1000:
                    last_chunk = {
                        "id": f"{article_id}_{chunk_index}",
                        "title": prefix or title,
                        "content": joined,
                        "contents": concat_title_and_content(prefix or title, joined),
                        "source": "statpearls",
                    }
                    chunks.append(last_chunk)
                    chunk_index += 1
                else:
                    for item_text in list_text:
                        chunks.append(
                            {
                                "id": f"{article_id}_{chunk_index}",
                                "title": prefix or title,
                                "content": item_text,
                                "contents": concat_title_and_content(prefix or title, item_text),
                                "source": "statpearls",
                            }
                        )
                        chunk_index += 1
                    last_chunk = None

                if last_node is not None and is_subtitle(last_node):
                    prefix = " -- ".join(part for part in [title, sec_title] if part)

            last_node = child

    return chunks


def download_file(url: str, output_path: Path) -> None:
    """Download the StatPearls archive with resume support."""
    downloaded_size = output_path.stat().st_size if output_path.exists() else 0
    headers = {"Range": f"bytes={downloaded_size}-"} if downloaded_size else {}

    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    append_mode = downloaded_size > 0 and response.status_code == 206
    total_size = int(response.headers.get("content-length", 0)) + (
        downloaded_size if append_mode else 0
    )

    with output_path.open("ab" if append_mode else "wb") as handle:
        with tqdm(
            total=total_size,
            initial=downloaded_size if append_mode else 0,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading StatPearls",
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract the downloaded archive."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in tqdm(members, desc="Extracting StatPearls"):
            archive.extract(member, path=output_dir)


def process_statpearls_directory(extracted_dir: Path, chunk_dir: Path) -> List[Dict]:
    """Process every NXML file into JSONL chunk files and return combined records."""
    chunk_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict] = []
    nxml_files = sorted(extracted_dir.glob("*.nxml"))
    for file_path in tqdm(nxml_files, desc="Processing StatPearls"):
        article_chunks = extract_statpearls_article(file_path)
        if not article_chunks:
            continue

        output_path = chunk_dir / f"{file_path.stem}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for chunk in article_chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        all_chunks.extend(article_chunks)

    return all_chunks


def build_statpearls_dataset(base_dir: Path) -> Dict[str, object]:
    """
    Download, extract, and convert StatPearls into reusable corpus files.

    The return payload is structured so callers can log or persist the same
    metadata without recomputing counts in each script.
    """
    statpearls_dir = base_dir / "statpearls"
    archive_path = statpearls_dir / STATPEARLS_ARCHIVE
    extracted_dir = statpearls_dir / EXTRACTED_DIR_NAME
    chunk_dir = statpearls_dir / "chunk"
    combined_path = statpearls_dir / "statpearls_combined.json"

    statpearls_dir.mkdir(parents=True, exist_ok=True)

    if not extracted_dir.exists() or len(list(extracted_dir.glob("*.nxml"))) < EXPECTED_NXML_COUNT:
        if not archive_path.exists():
            download_file(STATPEARLS_URL, archive_path)
        extract_archive(archive_path, statpearls_dir)

    chunks = process_statpearls_directory(extracted_dir, chunk_dir)
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)

    return {
        "statpearls_dir": str(statpearls_dir),
        "archive_path": str(archive_path),
        "extracted_dir": str(extracted_dir),
        "chunk_dir": str(chunk_dir),
        "combined_file": str(combined_path),
        "article_count": len(list(extracted_dir.glob("*.nxml"))),
        "chunk_count": len(chunks),
    }

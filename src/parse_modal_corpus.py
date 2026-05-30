#!/usr/bin/env python3
"""Flatten the MODAL TEI XML files into relation-level TSV rows.

The raw corpus stores transcript text in TEI body nodes and annotations in
separate Analec layers. This script joins the marker, scope, and relation
layers so downstream experiments can work with ordinary rows.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    value = value.replace("·", "")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def strip_ref(value: str | None) -> str:
    return (value or "").strip().lstrip("#")


def fs_properties(root: ET.Element) -> dict[str, dict[str, str]]:
    properties: dict[str, dict[str, str]] = {}
    for fs in root.findall(".//tei:fs", TEI_NS):
        fs_id = fs.get(XML_ID)
        if not fs_id:
            continue
        row: dict[str, str] = {}
        for feature in fs.findall("tei:f", TEI_NS):
            name = feature.get("name")
            text_node = feature.find("tei:string", TEI_NS)
            if name and text_node is not None:
                row[normalize_key(name)] = clean_text(text_node.text)
        properties[fs_id] = row
    return properties


TURN_WINDOWS = (1, 3, 5, 10, 20)


def body_text_and_anchors(root: ET.Element) -> tuple[str, dict[str, int], list[dict[str, int | str]]]:
    pieces: list[str] = []
    anchors: dict[str, int] = {}
    turns: list[dict[str, int | str]] = []
    cursor = 0

    def add(text: str | None) -> None:
        nonlocal cursor
        if text:
            pieces.append(text)
            cursor += len(text)

    def walk(element: ET.Element) -> None:
        add(element.text)
        for child in list(element):
            if child.tag.endswith("anchor"):
                anchor_id = child.get(XML_ID)
                if anchor_id:
                    anchors[anchor_id] = cursor
            else:
                walk(child)
            add(child.tail)

    for paragraph in root.findall(".//tei:text/tei:body/tei:p", TEI_NS):
        start = cursor
        walk(paragraph)
        end = cursor
        turn_text = clean_text("".join(pieces)[start:end])
        if turn_text:
            turns.append({"start": start, "end": end, "text": turn_text})
        add("\n")

    return "".join(pieces), anchors, turns


def span_text(text: str, anchors: dict[str, int], start_anchor: str, end_anchor: str) -> str:
    start = anchors.get(start_anchor)
    end = anchors.get(end_anchor)
    if start is None or end is None or end < start:
        return ""
    return clean_text(text[start:end])


def context_window(
    text: str,
    anchors: dict[str, int],
    start_anchor: str,
    end_anchor: str,
    width: int,
) -> str:
    start = anchors.get(start_anchor)
    end = anchors.get(end_anchor)
    if start is None or end is None:
        return ""
    return clean_text(text[max(0, start - width) : min(len(text), end + width)])


def turn_prefixes(text: str, turns: list[dict[str, int | str]], marker_start: int | None) -> dict[str, str]:
    if marker_start is None:
        return {f"turn_prefix_{window}": "" for window in TURN_WINDOWS}

    current_index = None
    for idx, turn in enumerate(turns):
        if int(turn["start"]) <= marker_start <= int(turn["end"]):
            current_index = idx
            break
    if current_index is None:
        return {f"turn_prefix_{window}": "" for window in TURN_WINDOWS}

    source_start_index = 0
    for idx in range(current_index, -1, -1):
        turn_text = str(turns[idx]["text"])
        if turn_text.startswith("MODAL"):
            source_start_index = idx
            break

    out = {}
    current_turn_start = int(turns[current_index]["start"])
    current_partial = clean_text(text[current_turn_start:marker_start])
    for window in TURN_WINDOWS:
        first_turn = max(source_start_index, current_index - window)
        previous_parts = [
            str(turns[idx]["text"])
            for idx in range(first_turn, current_index)
            if str(turns[idx]["text"])
        ]
        parts = previous_parts + ([current_partial] if current_partial else [])
        out[f"turn_prefix_{window}"] = clean_text("\n".join(parts))
    return out


def collect_spans(
    root: ET.Element,
    text: str,
    anchors: dict[str, int],
    properties: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    spans: dict[str, dict[str, str]] = {}
    for span_group in root.findall(".//tei:spanGrp", TEI_NS):
        layer = span_group.get("n") or ""
        for span in span_group.findall("tei:span", TEI_NS):
            span_id = span.get(XML_ID)
            if not span_id:
                continue
            start_anchor = strip_ref(span.get("from"))
            end_anchor = strip_ref(span.get("to"))
            start_char = anchors.get(start_anchor)
            end_char = anchors.get(end_anchor)
            ana = strip_ref(span.get("ana"))
            spans[span_id] = {
                "id": span_id,
                "layer": layer,
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "_start_char": str(start_char) if start_char is not None else "",
                "_end_char": str(end_char) if end_char is not None else "",
                "text": span_text(text, anchors, start_anchor, end_anchor),
                **properties.get(ana, {}),
            }
    return spans


def collect_scope_parts(root: ET.Element) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = {}
    for join_group in root.findall('.//tei:joinGrp[@n="scope"]', TEI_NS):
        for join in join_group.findall("tei:join", TEI_NS):
            scope_id = join.get(XML_ID)
            if not scope_id:
                continue
            scopes[scope_id] = [
                strip_ref(target)
                for target in (join.get("target") or "").split()
                if strip_ref(target)
            ]
    return scopes


def prefixed(prefix: str, values: dict[str, str]) -> dict[str, str]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalized_relation_fields(relation_props: dict[str, str]) -> dict[str, str]:
    age_group = relation_props.get("context_age_of_the_speakers") or relation_props.get("context_age_of_speakers", "")
    return {
        "relation_construction_function_norm": normalize_key(relation_props.get("construction_function", "")),
        "relation_construction_epistemic_type_norm": normalize_key(
            relation_props.get("construction_epistemic_type", "")
        ),
        "relation_construction_source_norm": normalize_key(relation_props.get("construction_source", "")),
        "relation_construction_polarity_norm": normalize_key(relation_props.get("construction_polarity", "")),
        "relation_context_age_group": age_group,
    }


def parse_modal_file(path: Path, context_chars: int) -> list[dict[str, str]]:
    language = path.stem.removeprefix("Modal-").removesuffix("-all")
    root = ET.parse(path).getroot()
    properties = fs_properties(root)
    text, anchors, turns = body_text_and_anchors(root)
    spans = collect_spans(root, text, anchors, properties)
    scope_parts = collect_scope_parts(root)

    rows: list[dict[str, str]] = []
    for join_group in root.findall('.//tei:joinGrp[@n="epistemic_relation"]', TEI_NS):
        for join in join_group.findall("tei:join", TEI_NS):
            relation_id = join.get(XML_ID) or ""
            targets = [strip_ref(target) for target in (join.get("target") or "").split()]
            marker_id = next((target for target in targets if target.startswith("u-marker")), "")
            scope_id = next((target for target in targets if target.startswith("s-scope")), "")

            marker = spans.get(marker_id, {})
            scope_ids = scope_parts.get(scope_id, [])
            scope_text = clean_text(" ".join(spans.get(part_id, {}).get("text", "") for part_id in scope_ids))
            scope_props = properties.get(f"{scope_id}-fs", {})
            relation_props = properties.get(strip_ref(join.get("ana")), {})
            marker_start = int_or_none(marker.get("_start_char"))
            marker_end = int_or_none(marker.get("_end_char"))
            scope_ends = [
                int_or_none(spans.get(part_id, {}).get("_end_char"))
                for part_id in scope_ids
            ]
            target_end_candidates = [marker_end] + [value for value in scope_ends if value is not None]
            target_end = max(value for value in target_end_candidates if value is not None) if any(
                value is not None for value in target_end_candidates
            ) else None

            row = {
                "language": language,
                "source_file": path.name,
                "relation_id": relation_id,
                "marker_id": marker_id,
                "scope_id": scope_id,
                "scope_part_ids": " ".join(scope_ids),
                "marker_text": marker.get("text", ""),
                "scope_text": scope_text,
                "prefix_context": clean_text(
                    text[max(0, marker_start - context_chars) : marker_start]
                    if marker_start is not None
                    else ""
                ),
                "target_from_marker": clean_text(
                    text[marker_start:target_end]
                    if marker_start is not None and target_end is not None and target_end >= marker_start
                    else ""
                ),
                "suffix_context": clean_text(
                    text[marker_end : min(len(text), marker_end + context_chars)]
                    if marker_end is not None
                    else ""
                ),
                **turn_prefixes(text, turns, marker_start),
                "context": context_window(
                    text,
                    anchors,
                    marker.get("start_anchor", ""),
                    marker.get("end_anchor", ""),
                    context_chars,
                ),
            }
            row.update(prefixed("marker", {k: v for k, v in marker.items() if k not in {"id", "layer", "start_anchor", "end_anchor", "text", "_start_char", "_end_char"}}))
            row.update(prefixed("scope", scope_props))
            row.update(prefixed("relation", relation_props))
            row.update(normalized_relation_fields(relation_props))
            rows.append(row)
    return rows


def write_rows(rows: Iterable[dict[str, str]], output_path: Path) -> int:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/modal_corpus/raw"),
        help="Directory containing Modal-*-all.xml files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_relations.tsv"),
        help="Relation-level TSV to write.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language names matching file stems, e.g. English French Italian.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=1000,
        help="Characters of transcript context on each side of the marker.",
    )
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("Modal-*-all.xml"))
    if args.languages:
        requested = set(args.languages)
        files = [path for path in files if path.stem.removeprefix("Modal-").removesuffix("-all") in requested]
    if not files:
        raise SystemExit(f"No Modal XML files found in {args.raw_dir}")

    rows: list[dict[str, str]] = []
    for path in files:
        rows.extend(parse_modal_file(path, args.context_chars))
    count = write_rows(rows, args.output)
    print(f"Wrote {count} modal relations to {args.output}")


if __name__ == "__main__":
    main()

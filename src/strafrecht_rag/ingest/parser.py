"""XML → ``Uitspraak``. All Rechtspraak XML quirks live here and nowhere else.

The document is ``<open-rechtspraak>`` with an RDF metadata block, an optional
``<inhoudsindicatie>`` (editorial summary) and a ``<uitspraak>`` (or ``<conclusie>``)
body built from DocBook-like ``section`` / ``title`` / ``para`` / ``parablock`` /
lists / tables. Namespaces vary per year; we match on local names only.

One unparsable document yields one ``ParseError`` record, never a crashed batch.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .model import Paragraaf, Sectie, Uitspraak

LABEL_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", re.S)
WS_RE = re.compile(r"\s+")
BLOCK_TAGS = {"para", "title", "listitem", "entry"}  # elements that contribute paragraphs
SKIP_TAGS = {"mediaobject", "imageobject", "imagedata", "footnote"}
HEADING_ROLES = {"bold", "underline"}
HEADING_MAX = 70
KOP_MAX_PARAS = 8  # more loose paragraphs than this before the first <section> is body text
PARSER_VERSION = "2"  # part of the index fingerprint: bump when parsing output changes


@dataclass(frozen=True)
class ParseError:
    bron: str
    fout: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element) -> str:
    """Flattened text of an element, emphasis included, whitespace normalized."""
    parts = []
    for node in element.iter():
        if _local(node.tag) in SKIP_TAGS:
            continue
        if node.text:
            parts.append(node.text)
        if node.tail and node is not element:
            parts.append(node.tail)
    return WS_RE.sub(" ", "".join(parts)).strip()


# ------------------------------------------------------------------ metadata
def _meta(root: ET.Element) -> dict:
    descriptions = [el for el in root.iter() if _local(el.tag) == "Description"]
    if not descriptions:
        raise ValueError("geen rdf:Description in document")
    head = descriptions[0]
    values: dict[str, list[str]] = {}
    for el in head:
        values.setdefault(_local(el.tag), []).append(_text(el))
    first = {k: v[0] for k, v in values.items() if v}
    ecli = first.get("identifier", "")
    if not ecli.startswith("ECLI:"):
        raise ValueError(f"geen ECLI gevonden (identifier={ecli!r})")
    if "date" not in first:
        raise ValueError("geen uitspraakdatum")
    references = [r for r in values.get("references", []) if r and "resourceIdentifier" not in r]
    return {
        "ecli": ecli,
        "instantie": first.get("creator", ""),
        "datum": date.fromisoformat(first["date"][:10]),
        "rechtsgebieden": [s for s in values.get("subject", []) if s],
        "procedures": [p for p in values.get("procedure", []) if p],
        "zaaknummer": first.get("zaaknummer", ""),
        "soort": first.get("type", "Uitspraak"),
        "wetsverwijzingen": references,
    }


# ------------------------------------------------------------------ body
def _split_label(text: str) -> tuple[str, str]:
    m = LABEL_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return "", text


def _title_parts(title_el: ET.Element) -> tuple[str, str]:
    """Split a section title into (number, text); the number may sit in a <nr> child."""
    nr = ""
    parts = [title_el.text or ""]
    for child in title_el:
        if _local(child.tag) == "nr":
            nr = _text(child).rstrip(".")
        else:
            parts.append(_text(child))
        parts.append(child.tail or "")
    text = WS_RE.sub(" ", "".join(parts)).strip()
    if not nr:
        nr, text = _split_label(text)
    return nr, text


def _blocks(element: ET.Element) -> Iterator[ET.Element]:
    """Depth-first block elements below ``element``, stopping at nested sections."""
    for child in element:
        tag = _local(child.tag)
        if tag == "section" or tag in SKIP_TAGS:
            continue
        if tag in BLOCK_TAGS:
            yield child
            continue
        yield from _blocks(child)


def _heading(block: ET.Element) -> str | None:
    """Rulings often mark sub-headings as a lone bold/underlined paragraph; detect those."""
    if _local(block.tag) != "para" or len(block) != 1 or _local(block[0].tag) != "emphasis":
        return None
    if (block.text or "").strip() or (block[0].tail or "").strip():
        return None
    if block[0].attrib.get("role") not in HEADING_ROLES:
        return None
    text = _text(block[0]).rstrip(".:")
    letters = sum(ch.isalpha() for ch in text)
    if not 4 <= len(text) <= HEADING_MAX or letters < 4 or text.startswith("["):
        return None
    return text


def _section_from(children: list[ET.Element], nummer: str, titel: str) -> list[Sectie]:
    """Paragraphs under one section; lone bold headings split it into sub-sections."""
    out: list[Sectie] = []
    paragrafen: list[Paragraaf] = []
    current_nr, current_titel = nummer, titel
    headings = 0

    def flush() -> None:
        if paragrafen:
            out.append(Sectie(nummer=current_nr, titel=current_titel, paragrafen=list(paragrafen)))
        paragrafen.clear()

    for el in children:
        blocks = [el] if _local(el.tag) in BLOCK_TAGS else list(_blocks(el))
        for block in blocks:
            if (heading := _heading(block)) is not None:
                flush()
                headings += 1
                current_nr, current_titel = f"{nummer}.{headings}", f"{titel} › {heading}"
                continue
            text = _text(block)
            if not text:
                continue
            label, _ = _split_label(text)
            paragrafen.append(Paragraaf(nummer=len(paragrafen) + 1, tekst=text, label=label))
    flush()
    return out


def _walk_sections(container: ET.Element, secties: list[Sectie]) -> None:
    """Top-level body: numbered sections, plus loose text before/after them."""
    has_sections = any(_local(c.tag) == "section" for c in container)
    ordinal = 0
    loose: list[ET.Element] = []

    def add_loose(titel: str) -> None:
        nonlocal ordinal
        produced = _section_from(loose, str(ordinal + 1), titel)
        if titel == "Kop" and sum(len(sec.paragrafen) for sec in produced) > KOP_MAX_PARAS:
            produced = _section_from(loose, str(ordinal + 1), "Uitspraak")
        if produced:
            ordinal += 1
            secties.extend(produced)
        loose.clear()

    for child in container:
        if _local(child.tag) == "section":
            if loose:
                add_loose("Kop")
            ordinal += 1
            _add_section(child, secties, "", ordinal)
        else:
            loose.append(child)
    if loose:
        add_loose("Uitspraak" if not has_sections else "Slot")


def _add_section(section: ET.Element, secties: list[Sectie], parent_nr: str, ordinal: int) -> None:
    title_el = next((c for c in section if _local(c.tag) == "title"), None)
    nr, titel = _title_parts(title_el) if title_el is not None else ("", "")
    if not nr:
        nr = f"{parent_nr}.{ordinal}" if parent_nr else str(ordinal)
    body = [c for c in section if c is not title_el and _local(c.tag) != "section"]
    nested = [c for c in section if _local(c.tag) == "section"]
    secties.extend(_section_from(body, nr, titel or f"Sectie {nr}"))
    for i, sub in enumerate(nested, 1):
        _add_section(sub, secties, nr, i)


def parse_xml(payload: bytes | str) -> Uitspraak:
    root = ET.fromstring(payload)
    meta = _meta(root)
    summary_el = next((el for el in root.iter() if _local(el.tag) == "inhoudsindicatie"), None)
    samenvatting = _text(summary_el) if summary_el is not None else ""
    body = next((el for el in root.iter() if _local(el.tag) in {"uitspraak", "conclusie"}), None)
    if body is None:
        raise ValueError("geen <uitspraak> of <conclusie> body")
    secties: list[Sectie] = []
    if samenvatting:
        secties.append(
            Sectie(
                nummer="0",
                titel="Inhoudsindicatie",
                paragrafen=[Paragraaf(nummer=1, tekst=samenvatting)],
            )
        )
    _walk_sections(body, secties)
    if len(secties) <= (1 if samenvatting else 0):
        raise ValueError("body zonder tekst")
    return Uitspraak(samenvatting=samenvatting, secties=secties, **meta)


def parse_file(path: Path) -> Uitspraak | ParseError:
    try:
        return parse_xml(path.read_bytes())
    except (ET.ParseError, ValueError, KeyError) as exc:
        return ParseError(bron=path.name, fout=f"{type(exc).__name__}: {exc}")


def parse_dir(raw_dir: Path) -> Iterator[Uitspraak | ParseError]:
    for path in sorted(raw_dir.glob("*.xml")):
        yield parse_file(path)

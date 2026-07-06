from __future__ import annotations

import re
from typing import Any


def extract_reference_titles(report: str) -> list[str]:
    match = re.search(r"references\s*:?\s*(.*)$", report or "", flags=re.I | re.S)
    if not match:
        return []
    titles = []
    for line in match.group(1).splitlines():
        title = line.strip().strip("-*0123456789.、 ")
        if title:
            titles.append(title)
    return titles


def verify_citations(report: str, retrieved_papers: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_titles = [str(paper.get("title") or "").strip() for paper in retrieved_papers if paper.get("title")]
    allowed_lower = {title.lower() for title in allowed_titles}
    references = extract_reference_titles(report)
    if not references:
        report_lower = (report or "").lower()
        references = [title for title in allowed_titles if title.lower() in report_lower]

    unsupported = [title for title in references if title.lower() not in allowed_lower]
    supported = [title for title in references if title.lower() in allowed_lower]
    return {
        "total_references": len(references),
        "supported_references": len(supported),
        "unsupported_titles": unsupported,
        "verification_passed": not unsupported,
    }

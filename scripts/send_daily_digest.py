#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any


CORPUS_PATH = Path("web/data/rag_corpus.json")


def load_top_papers(limit: int = 5) -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        print(f"RAG corpus not found: {CORPUS_PATH}")
        return []
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    papers = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    return sorted(papers, key=lambda item: float(item.get("final_score") or 0.0), reverse=True)[:limit]


def build_digest(papers: list[dict[str, Any]]) -> str:
    lines = ["AI-PaperFlow Daily Top Papers", ""]
    if not papers:
        lines.append("No papers available.")
        return "\n".join(lines)
    for index, paper in enumerate(papers, start=1):
        summary = paper.get("problem") or paper.get("abstract") or paper.get("summary") or ""
        lines.extend(
            [
                f"{index}. {paper.get('title', '')}",
                f"Final Score: {float(paper.get('final_score') or 0.0):.2f}",
                f"Summary: {summary}",
                f"PDF: {paper.get('pdf_url', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def smtp_config() -> dict[str, str]:
    return {
        "host": os.getenv("SMTP_HOST", "").strip(),
        "port": os.getenv("SMTP_PORT", "587").strip(),
        "user": os.getenv("SMTP_USER", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip(),
        "to": os.getenv("DIGEST_TO", "").strip(),
    }


def send_email(body: str) -> bool:
    config = smtp_config()
    if not all([config["host"], config["port"], config["user"], config["password"], config["to"]]):
        print("SMTP config is incomplete. Printing digest instead.")
        print(body)
        return False

    message = EmailMessage()
    message["Subject"] = "AI-PaperFlow Daily Top Papers"
    message["From"] = config["user"]
    message["To"] = config["to"]
    message.set_content(body)

    with smtplib.SMTP(config["host"], int(config["port"]), timeout=30) as server:
        server.starttls()
        server.login(config["user"], config["password"])
        server.send_message(message)
    return True


def main() -> None:
    digest = build_digest(load_top_papers())
    try:
        sent = send_email(digest)
    except Exception as exc:
        print(f"SMTP send failed. Printing digest instead: {exc}")
        print(digest)
        return
    print("Daily digest sent." if sent else "Daily digest printed.")


if __name__ == "__main__":
    main()

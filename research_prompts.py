from __future__ import annotations

import json


DEFAULT_SUB_QUESTIONS = [
    "What is the research background and motivation?",
    "What are the main technical methods?",
    "Which key papers are most relevant?",
    "What limitations are reported or implied?",
    "What future directions are suggested?",
]


def planner_prompt(topic: str) -> str:
    return f"""
请将研究主题拆解为 3 到 5 个用于论文调研的子问题。
子问题需要覆盖 research background、main methods、key papers、limitations、future directions。
只输出 JSON 数组，不要输出额外解释。

研究主题：{topic}
""".strip()


def writer_prompt(topic: str, sub_questions: list[str], evidence_pack: list[dict]) -> str:
    evidence_json = json.dumps(evidence_pack, ensure_ascii=False, indent=2)
    sub_question_text = "\n".join(f"- {item}" for item in sub_questions)
    return f"""
你是一个严谨的中文论文调研助手。请只根据给定 evidence_pack 生成结构化调研报告。
不要编造 evidence_pack 中不存在的论文标题、实验结果或结论。

研究主题：
{topic}

子问题：
{sub_question_text}

evidence_pack:
{evidence_json}

报告结构必须包含以下标题：
1. Research Background
2. Key Findings
3. Main Methods
4. Paper Comparison
5. Limitations
6. Future Directions
7. References

References 中只能列出 evidence_pack 里出现过的论文标题。
请使用中文回答，保留上述英文小节标题。
""".strip()

#!/usr/bin/env python3
"""Convert display math blocks in the paper to Markdown-friendly text blocks.

Cursor/VSCode Markdown preview in this workspace is not rendering $$...$$ as
math. Keeping LaTeX display equations therefore produces unreadable preview.
This script converts display equations into fenced text blocks so the Markdown
preview remains stable and readable.
"""
from __future__ import annotations

import re
from pathlib import Path


PAPER = Path(__file__).resolve().parent.parent / "paper" / "FCGJ-v3.0.md"


def cleanup_formula(src: str) -> str:
    """Make a LaTeX-ish equation block readable as plain Markdown text."""
    text = src.strip()

    replacements = {
        r"\begin{aligned}": "",
        r"\end{aligned}": "",
        r"\begin{bmatrix}": "[",
        r"\end{bmatrix}": "]",
        r"\top": "T",
        r"\left[": "[",
        r"\right]": "]",
        r"\left(": "(",
        r"\right)": ")",
        r"\left\{": "{",
        r"\right\}": "}",
        r"\qquad": "    ",
        r"\quad": "  ",
        r"\,": " ",
        r"\;": " ",
        r"\\": "\n",
        r"\mathbf": "",
        r"\boldsymbol": "",
        r"\mathcal": "",
        r"\mathrm": "",
        r"\operatorname": "",
        r"\text": "",
        r"\hat": "hat",
        r"\bar": "bar",
        r"\sqrt": "sqrt",
        r"\frac": "frac",
        r"\sum": "sum",
        r"\min": "min",
        r"\max": "max",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\arctan": "arctan",
        r"\arctan2": "arctan2",
        r"\arccos": "arccos",
        r"\exp": "exp",
        r"\log": "log",
        r"\epsilon": "eps",
        r"\Delta": "Delta",
        r"\Theta": "Theta",
        r"\lambda": "lambda",
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\eta": "eta",
        r"\rho": "rho",
        r"\omega": "omega",
        r"\sigma": "sigma",
        r"\phi": "phi",
        r"\theta": "theta",
        r"\psi": "psi",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove LaTeX grouping braces in common commands while keeping subscripts readable.
    text = text.replace("{", "").replace("}", "")
    text = text.replace("&=", "=").replace("&", " ")
    text = text.replace("^T", "^T")

    # Collapse excessive blank lines caused by row separators.
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line or (cleaned and cleaned[-1]):
            cleaned.append(line)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        formula = cleanup_formula(match.group(1))
        return f"\n```text\n{formula}\n```\n"

    text = re.sub(r"\n?\$\$\n(.*?)\n\$\$\n?", repl, text, flags=re.S)
    PAPER.write_text(text, encoding="utf-8")
    print(f"Converted display math to Markdown text blocks: {PAPER}")


if __name__ == "__main__":
    main()

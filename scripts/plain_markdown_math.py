#!/usr/bin/env python3
"""Convert remaining inline math and equation text to plain Markdown.

This project paper is currently edited as Markdown, not LaTeX. To make the
Cursor Markdown Preview readable without math rendering, this script converts:

- inline `$...$` math into code spans, e.g. `lambda_dir`
- already fenced equation text blocks into cleaner ASCII-style pseudo formulas
"""
from __future__ import annotations

import re
from pathlib import Path


PAPER = Path(__file__).resolve().parent.parent / "paper" / "FCGJ-v3.0.md"

GREEK = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\Delta": "Delta",
    r"\epsilon": "eps",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\Theta": "Theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\omega": "omega",
    r"\phi": "phi",
    r"\psi": "psi",
    r"\rho": "rho",
    r"\sigma": "sigma",
}

OPS = {
    r"\rightarrow": "->",
    r"\leq": "<=",
    r"\geq": ">=",
    r"\approx": "~",
    r"\times": "x",
    r"\in": " in ",
    r"\pm": "+/-",
    r"\cdot": "*",
    r"\operatorname": "",
    r"\arctan2": "atan2",
    r"\arctan": "atan",
    r"\arccos": "acos",
    r"\cos": "cos",
    r"\sin": "sin",
    r"\sqrt": "sqrt",
    r"\sum": "sum",
    r"\min": "min",
    r"\max": "max",
    r"\log": "log",
    r"\exp": "exp",
    r"\frac": "frac",
    r"\tfrac": "frac",
    r"\left": "",
    r"\right": "",
    r"\top": "T",
    r"\_": "_",
}


def strip_latex_wrappers(s: str) -> str:
    # Replace commands with one braced argument: \mathbf{x} -> x
    pattern = re.compile(r"\\(?:mathbf|mathcal|boldsymbol|mathrm|text|hat|bar)\{([^{}]+)\}")
    old = None
    while old != s:
        old = s
        s = pattern.sub(r"\1", s)
    return s


def to_plain_math(s: str) -> str:
    s = strip_latex_wrappers(s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    for k, v in OPS.items():
        s = s.replace(k, v)

    # Common leftovers after previous conversion.
    fixes = {
        r"\mathbbR": "R",
        r"\mathbbE": "E",
        r"\|": "||",
        r"\{": "{",
        r"\}": "}",
        r"\[": "[",
        r"\]": "]",
        r"\(": "(",
        r"\)": ")",
        r"\begincases": "cases:",
        r"\endcases": "",
        "fraceE_wu": "(e / E_wu)",
        "frac1Nsum_k=1^N": "(1 / N) * sum_{k=1..N}",
        "frac12sum_i": "0.5 * sum_i",
        "frace_i^2sigma_i^2": "(e_i^2 / sigma_i^2)",
        "fracbarnn_b(k)": "(n_bar / n_b(k))",
        "frac1W_max": "(1 / W_max)",
        "fracmin(dirMAE,90^circ)30^circ": "min(dirMAE, 90 deg) / 30 deg",
        "frac||w_h,k||-w_min": "(||w_h,k|| - w_min) /",
        "frac||w_h,k||w_s": "||w_h,k|| / w_s",
        "frac||w_h,k||_22w_min": "||w_h,k||_2 / (2*w_min)",
        "fracRMSE_h(b)": "RMSE_h(b) /",
    }
    for k, v in fixes.items():
        s = s.replace(k, v)

    # Remove remaining LaTeX-only braces while preserving simple expression readability.
    s = s.replace("{", "").replace("}", "")
    s = s.replace("^circ", " deg")
    s = s.replace("\\", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_fenced_blocks(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        lines = [to_plain_math(line) for line in body.splitlines()]
        # Keep intentional line breaks, but drop repeated empty lines.
        cleaned: list[str] = []
        for line in lines:
            if line or (cleaned and cleaned[-1]):
                cleaned.append(line)
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        return "```text\n" + "\n".join(cleaned) + "\n```"

    return re.sub(r"```text\n(.*?)\n```", repl, text, flags=re.S)


def clean_inline_math(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = to_plain_math(match.group(1))
        return f"`{inner}`"

    # Do not cross line boundaries. This avoids touching tables or malformed blocks too broadly.
    return re.sub(r"(?<!\\)\$([^$\n]+?)\$", repl, text)


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")
    text = clean_fenced_blocks(text)
    text = clean_inline_math(text)
    PAPER.write_text(text, encoding="utf-8")
    print(f"Converted inline math to plain Markdown in {PAPER}")


if __name__ == "__main__":
    main()

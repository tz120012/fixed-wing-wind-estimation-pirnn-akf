"""Figure 1 - pure web-code (HTML + inline SVG) rendering.

This is a second, independent renderer for the same architecture diagram as
`plot_figure1_architecture.py` (Matplotlib/PNG/PDF/SVG-for-print), but emits
a single self-contained `.html` file built from native SVG primitives
(<rect>, <path>, <text>/<tspan>) plus a small amount of CSS -- i.e. it can be
opened directly in any browser, with real selectable/searchable text, no
Python/Matplotlib dependency at view time, and no external assets or network
calls.

It intentionally reuses the exact same layout coordinates, box sizes, and
routing logic as `plot_figure1_architecture.py` (same coordinate system,
same DX shift for the onboard-sensors box) so the two renderings stay
geometrically consistent. Math labels are approximated with Unicode
sub/superscripts and bold/italic <tspan> runs instead of LaTeX, since
browsers have no native TeX renderer.

Run:
    python scripts/render_figure1_web.py

Output:
    paper/figures/figure1_web.html
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLUE = "#1F77B4"
RED = "#E64B35"
GREEN = "#00A087"
GRAY = "#7F7F7F"
BOX_EDGE = "#595959"
STAGE_EDGE = "#8C8C8C"
STAGE_FILL = "#F3F7FB"

# ---- coordinate transform: same data window as plot_figure1_architecture.py ----
X0, X1 = -0.7, 26.4
Y0, Y1 = -6.15, 3.85
SCALE = 42.0
W_PX = (X1 - X0) * SCALE
H_PX = (Y1 - Y0) * SCALE


def tx(x: float) -> float:
    return (x - X0) * SCALE


def ty(y: float) -> float:
    return (Y1 - y) * SCALE


svg_parts: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rounded_box(cx, cy, w, h, *, edge=BOX_EDGE, fill="white", lw=1.4, dash=None, r=6):
    x, y = tx(cx - w / 2), ty(cy + h / 2)
    ww, hh = w * SCALE, h * SCALE
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    svg_parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{ww:.1f}" height="{hh:.1f}" rx="{r}" ry="{r}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="{lw}"{dash_attr}/>'
    )


def stage_box(x0, x1, y0, y1, title):
    x, y = tx(x0), ty(y1)
    ww, hh = (x1 - x0) * SCALE, (y1 - y0) * SCALE
    svg_parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{ww:.1f}" height="{hh:.1f}" rx="9" ry="9" '
        f'fill="{STAGE_FILL}" stroke="{STAGE_EDGE}" stroke-width="1.2"/>'
    )
    svg_parts.append(
        f'<text x="{x + 10:.1f}" y="{y + 17:.1f}" class="stage-title">{esc(title)}</text>'
    )


def label(cx, cy, lines, *, size=12.2, weight="normal"):
    """lines: list of plain/marked-up strings; markup uses ^{..} for
    superscript and _{..} for subscript, and *..* for bold-italic vectors."""
    x = tx(cx)
    n = len(lines)
    line_h = size * 1.32
    y_start = ty(cy) - (n - 1) * line_h / 2 + size * 0.35
    tspans = []
    for i, line in enumerate(lines):
        y = y_start + i * line_h
        tspans.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                       f'class="lbl" font-size="{size}" font-weight="{weight}">{_markup(line)}</text>')
    svg_parts.extend(tspans)


def _markup(s: str) -> str:
    """Very small inline markup: **bold**, ~sub~, ^sup^.

    Uses relative `dy` (em) shifts rather than `baseline-shift` (which some
    SVG rasterizers, e.g. cairosvg, mis-render as huge fallback glyphs for
    percentage values). Each run's dy is the *delta* from the running
    baseline offset so subscripts/superscripts snap back to the normal
    baseline once a plain-text run follows.
    """
    out = []
    i = 0
    shift = 0.0  # current cumulative baseline offset, in em (+down / -up)
    SUB, SUP = 0.30, -0.34

    def emit(text: str, target_shift: float, size_pct: int | None):
        nonlocal shift
        dy = target_shift - shift
        shift = target_shift
        attrs = ""
        if abs(dy) > 1e-6:
            attrs += f' dy="{dy:.2f}em"'
        if size_pct is not None:
            attrs += f' font-size="{size_pct}%"'
        out.append(f'<tspan{attrs}>{esc(text)}</tspan>')

    while i < len(s):
        if s[i:i + 2] == "**":
            j = s.index("**", i + 2)
            emit(s[i + 2:j], 0.0, None)
            out[-1] = out[-1].replace("<tspan", '<tspan font-weight="700"', 1)
            i = j + 2
        elif s[i] == "~":
            j = s.index("~", i + 1)
            emit(s[i + 1:j], SUB, 72)
            i = j + 1
        elif s[i] == "^":
            j = s.index("^", i + 1)
            emit(s[i + 1:j], SUP, 72)
            i = j + 1
        else:
            j = i
            while j < len(s) and s[j] not in "*~^":
                j += 1
            emit(s[i:j], 0.0, None)
            i = j
    return "".join(out)


def elbow(points, *, color, dash=None, lw=1.5):
    d = f"M {tx(points[0][0]):.1f} {ty(points[0][1]):.1f} " + " ".join(
        f"L {tx(x):.1f} {ty(y):.1f}" for x, y in points[1:]
    )
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker = f"arrow-{color.lstrip('#')}"
    svg_parts.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{lw}"{dash_attr} '
        f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#{marker})"/>'
    )


def arrow_markers():
    parts = []
    for c in {BLUE, RED, GREEN, GRAY}:
        mid = f"arrow-{c.lstrip('#')}"
        parts.append(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        )
    return "".join(parts)


def build():
    DX = 3.1
    BOX_W, BOX_H = 2.35, 0.95
    HEAD_W, HEAD_H = 2.7, 0.80

    stage_box(-0.5, 11.9 + DX, -2.35, 3.65, "PI-GRU front-end")
    stage_box(12.35 + DX, 23.1 + DX, -2.35, 3.65, "AKF back-end")
    stage_box(-0.5, 23.1 + DX, -5.65, -2.65, "Training constraints / kinematics")

    # onboard sensors -> trunk
    x_sens, y_sens = 1.15, 0.9
    SENS_W = 2.55
    rounded_box(x_sens, y_sens, SENS_W, BOX_H)
    label(x_sens, y_sens, ["Onboard sensors", "GPS/IMU/pitot/AHRS"], size=11.6)

    x_in, x_gru, x_lat = 1.15 + DX, 4.05 + DX, 6.95 + DX
    y_trunk = 0.9
    rounded_box(x_in, y_trunk, BOX_W, BOX_H)
    label(x_in, y_trunk, ["Sliding window", "41-D, z-norm", "**X**~k-T+1:k~"], size=10.6)
    rounded_box(x_gru, y_trunk, BOX_W, BOX_H)
    label(x_gru, y_trunk, ["GRU backbone", "temporal feature"], size=11.8)
    rounded_box(x_lat, y_trunk, BOX_W, BOX_H)
    label(x_lat, y_trunk, ["Shared latent", "**z**~k~"], size=12.0)

    elbow([(x_sens + SENS_W / 2, y_sens), (x_in - BOX_W / 2, y_trunk)], color=BLUE)
    elbow([(x_in + BOX_W / 2, y_trunk), (x_gru - BOX_W / 2, y_trunk)], color=BLUE)
    elbow([(x_gru + BOX_W / 2, y_trunk), (x_lat - BOX_W / 2, y_trunk)], color=BLUE)

    # five output heads
    x_head = 9.95 + DX
    y_wind, y_r, y_conf, y_q, y_ang = 2.85, 1.70, 0.55, -0.60, -1.75

    rounded_box(x_head, y_wind, HEAD_W, HEAD_H, edge=BLUE, lw=1.6)
    label(x_head, y_wind, ["Wind head", "**w**~NN,k~"], size=11.8)
    rounded_box(x_head, y_r, HEAD_W, HEAD_H, edge=BLUE, lw=1.6)
    label(x_head, y_r, ["Dynamic-R head", "**r**~scale,k~"], size=11.8)
    rounded_box(x_head, y_conf, HEAD_W, HEAD_H, edge=BLUE, lw=1.6)
    label(x_head, y_conf, ["Confidence head", "s~k~"], size=11.8)
    rounded_box(x_head, y_q, HEAD_W, HEAD_H, edge=GRAY, lw=1.4, dash="6,4")
    label(x_head, y_q, ["Diagnostic-Q head", "**q**~scale,k~"], size=11.8)
    rounded_box(x_head, y_ang, HEAD_W, HEAD_H, edge=BLUE, lw=1.6)
    label(x_head, y_ang, ["Angle / TAS-scale head", "[\u0394\u03b1, \u0394\u03b2, s~TAS~]"], size=10.8)

    x_split = x_lat + BOX_W / 2 + 0.55
    for y_h, c, dash in [(y_wind, BLUE, None), (y_r, BLUE, None), (y_conf, BLUE, None), (y_ang, BLUE, None)]:
        elbow([(x_lat + BOX_W / 2, y_trunk), (x_split, y_trunk), (x_split, y_h),
               (x_head - HEAD_W / 2, y_h)], color=c, dash=dash)
    elbow([(x_lat + BOX_W / 2, y_trunk), (x_split, y_trunk), (x_split, y_q),
           (x_head - HEAD_W / 2, y_q)], color=GRAY, dash="6,4")

    # AKF back-end
    x_akf, y_akf = 15.55 + DX, 1.05
    AKF_W, AKF_H = 3.05, 1.65
    rounded_box(x_akf, y_akf, AKF_W, AKF_H, edge=GREEN, lw=1.8)
    label(x_akf, y_akf,
          ["Dynamic-R AKF", "prediction + update",
           "R~k~ = D~r,k~ R~0~ D~r,k~^T^", "x\u0302~k|k~"], size=10.8)

    y_wind_entry = y_akf + 0.32
    elbow([(x_head + HEAD_W / 2, y_wind), (x_akf - AKF_W / 2 - 0.4, y_wind),
           (x_akf - AKF_W / 2 - 0.4, y_wind_entry), (x_akf - AKF_W / 2, y_wind_entry)], color=GREEN)
    elbow([(x_head + HEAD_W / 2, y_r), (x_akf - AKF_W / 2, y_r)], color=GREEN)
    elbow([(x_head + HEAD_W / 2, y_q), (x_akf - AKF_W / 2 - 0.4, y_q),
           (x_akf - AKF_W / 2 - 0.4, y_akf - 0.6), (x_akf - AKF_W / 2, y_akf - 0.6)],
          color=GRAY, dash="6,4")

    x_ema, y_ema = 19.55 + DX, y_akf
    EMA_W, EMA_H = 2.85, 1.65
    rounded_box(x_ema, y_ema, EMA_W, EMA_H, edge=GREEN, lw=1.8)
    label(x_ema, y_ema,
          ["Confidence-gated EMA", "\u03c9~NN,k~ = clip(s~k~)",
           "**w**~AKF,k~ = \u03c9 x\u0302~k|k~", "+ (1-\u03c9) **w**~AKF,k-1~"], size=10.4)

    elbow([(x_akf + AKF_W / 2, y_akf), (x_ema - EMA_W / 2, y_ema)], color=GREEN)
    y_conf_bus = y_akf - AKF_H / 2 - 0.4
    x_conf_drop = x_akf - AKF_W / 2 - 1.15
    elbow([(x_head + HEAD_W / 2, y_conf), (x_conf_drop, y_conf), (x_conf_drop, y_conf_bus),
           (x_ema - EMA_W / 2 - 0.55, y_conf_bus), (x_ema - EMA_W / 2 - 0.55, y_ema - 0.5),
           (x_ema - EMA_W / 2, y_ema - 0.5)], color=BLUE)

    x_delay, y_delay = x_ema, y_ema - 1.55
    rounded_box(x_delay, y_delay, 2.15, 0.6, edge=GREEN, lw=1.3, dash="6,4")
    label(x_delay, y_delay, ["Delay: **w**~AKF,k-1~"], size=10.6)
    elbow([(x_ema - 0.6, y_ema - EMA_H / 2), (x_ema - 0.6, y_delay + 0.3)], color=GREEN, dash="6,4")
    elbow([(x_ema + 0.6, y_delay + 0.3), (x_ema + 0.6, y_ema - EMA_H / 2)], color=GREEN, dash="6,4")

    x_out, y_out = 22.35 + DX, y_ema
    OUT_W = 1.55
    rounded_box(x_out, y_out, OUT_W, 1.05, edge=GREEN, lw=1.8)
    label(x_out, y_out, ["Posterior", "wind", "w\u0302~AKF,k~"], size=10.8)
    elbow([(x_ema + EMA_W / 2, y_ema), (x_out - OUT_W / 2, y_out)], color=GREEN)

    # training constraints / kinematics
    y_train = -3.75
    KBOX_W, KBOX_H = 2.9, 0.95
    x_phys, x_data, x_weak, x_reg, x_kin = (
        2.15 + DX, 6.35 + DX, 10.55 + DX, 14.75 + DX, 18.95 + DX,
    )

    rounded_box(x_phys, y_train, KBOX_W, KBOX_H, edge=RED, lw=1.5)
    label(x_phys, y_train, ["Physics closure loss", "airspeed consistency"], size=11.2)
    rounded_box(x_data, y_train, KBOX_W, KBOX_H, edge=RED, lw=1.5)
    label(x_data, y_train, ["Data loss", "wind supervision"], size=11.2)
    rounded_box(x_weak, y_train, KBOX_W, KBOX_H, edge=RED, lw=1.5)
    label(x_weak, y_train, ["Weak-wind regularization", "hinge + SNR-aware weight"], size=10.6)
    rounded_box(x_reg, y_train, KBOX_W, KBOX_H, edge=RED, lw=1.5)
    label(x_reg, y_train, ["Uncertainty-scale reg.", "Huber + boundary on **q**,**r**~scale~"], size=10.4)
    rounded_box(x_kin, y_train, KBOX_W, KBOX_H, edge=BOX_EDGE, lw=1.4)
    label(x_kin, y_train, ["Kinematic slice", "**v**~g~, TAS, attitude"], size=11.2)

    y_bus_top = -2.55
    y_bus_bot = -5.25

    elbow([(x_in, y_trunk - BOX_H / 2), (x_in, y_bus_bot), (x_kin - 0.4, y_bus_bot),
           (x_kin - 0.4, y_train - KBOX_H / 2)], color=RED, dash="6,4")
    elbow([(x_kin + 0.4, y_train - KBOX_H / 2), (x_kin + 0.4, y_bus_bot), (x_phys, y_bus_bot),
           (x_phys, y_train - KBOX_H / 2)], color=RED, dash="6,4")

    x_wind_lane = x_head - HEAD_W / 2 - 0.35
    elbow([(x_head - HEAD_W / 2, y_wind), (x_wind_lane, y_wind), (x_wind_lane, y_bus_top),
           (x_phys + 0.55, y_bus_top), (x_phys + 0.55, y_train + KBOX_H / 2)], color=RED, dash="6,4")
    elbow([(x_wind_lane, y_bus_top), (x_data, y_bus_top), (x_data, y_train + KBOX_H / 2)],
          color=RED, dash="6,4")
    elbow([(x_wind_lane, y_bus_top), (x_weak - 0.55, y_bus_top), (x_weak - 0.55, y_train + KBOX_H / 2)],
          color=RED, dash="6,4")

    x_ang_lane = x_head - HEAD_W / 2 - 0.7
    elbow([(x_head - HEAD_W / 2, y_ang), (x_ang_lane, y_ang), (x_ang_lane, y_bus_top + 0.22),
           (x_phys + 0.9, y_bus_top + 0.22), (x_phys + 0.9, y_train + KBOX_H / 2)], color=RED, dash="6,4")

    x_q_lane = x_head - HEAD_W / 2 - 0.4
    elbow([(x_head - HEAD_W / 2, y_q), (x_q_lane, y_q), (x_q_lane, y_bus_top - 0.15),
           (x_reg - 0.5, y_bus_top - 0.15), (x_reg - 0.5, y_train + KBOX_H / 2)], color=RED, dash="6,4")
    x_r_lane = x_head + HEAD_W / 2 + 0.6
    elbow([(x_head + HEAD_W / 2, y_r), (x_r_lane, y_r), (x_r_lane, y_bus_top - 0.32),
           (x_reg + 0.5, y_bus_top - 0.32), (x_reg + 0.5, y_train + KBOX_H / 2)], color=RED, dash="6,4")

    # legend
    legend_items = [
        (BLUE, None, "inference path"),
        (RED, "6,4", "training-only loss path"),
        (GREEN, None, "AKF / EMA fusion"),
        (GRAY, "6,4", "diagnostic / optional signal"),
    ]
    lx0, ly0 = -0.3, -5.95
    for i, (c, dash, txt) in enumerate(legend_items):
        xs = lx0 + i * 5.6
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg_parts.append(
            f'<line x1="{tx(xs):.1f}" y1="{ty(ly0):.1f}" x2="{tx(xs + 0.55):.1f}" y2="{ty(ly0):.1f}" '
            f'stroke="{c}" stroke-width="2.2"{dash_attr}/>'
        )
        svg_parts.append(
            f'<text x="{tx(xs + 0.72):.1f}" y="{ty(ly0) + 4:.1f}" class="legend">{esc(txt)}</text>'
        )


def render_html() -> str:
    build()
    svg_body = "\n".join(svg_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Figure 1 - PI-GRU + PIRNN-AKF architecture</title>
<style>
  body {{ margin: 0; padding: 24px; background: #ffffff;
         font-family: "Times New Roman", Times, "Noto Serif", serif; }}
  .caption {{ max-width: {W_PX:.0f}px; font-size: 14px; color: #222; margin-top: 14px; line-height: 1.5; }}
  svg {{ display: block; max-width: 100%; height: auto; }}
  text.stage-title {{ font-size: 15px; fill: #222; font-weight: 600; }}
  text.lbl {{ fill: #111; }}
  text.legend {{ font-size: 13px; fill: #222; }}
</style>
</head>
<body>
<svg viewBox="0 0 {W_PX:.1f} {H_PX:.1f}" xmlns="http://www.w3.org/2000/svg">
  <defs>{arrow_markers()}</defs>
  <rect x="0" y="0" width="{W_PX:.1f}" height="{H_PX:.1f}" fill="white"/>
  {svg_body}
</svg>
<p class="caption"><b>Figure 1.</b> PI-GRU front-end + PIRNN-AKF back-end architecture
(web/SVG rendering, geometrically matched to <code>paper/figures/figure1.png</code> /
<code>scripts/plot_figure1_architecture.py</code>). Blue = inference path; red dashed =
training-only loss path; green = AKF/EMA fusion path; gray dashed = diagnostic/optional
signal. Math labels are approximated with Unicode sub/superscripts rather than LaTeX.</p>
</body>
</html>
"""


def main():
    html = render_html()
    out = PROJECT_ROOT / "paper" / "figures" / "figure1_web.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

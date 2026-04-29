"""
Shared visualization helpers used across all chapters.

Provides functions to render:
- ASTs and parse trees (graphviz)
- Control flow graphs (graphviz)
- Computation graphs / DAGs (graphviz)
- Heatmaps for memory access patterns (matplotlib)
- Side-by-side before/after comparisons (matplotlib)
- Bar charts for benchmarks (matplotlib)
"""

import os
import subprocess
import tempfile
from typing import Any

# ---------------------------------------------------------------------------
# Graphviz helpers
# ---------------------------------------------------------------------------

def graphviz_available() -> bool:
    """Check if the graphviz 'dot' binary is installed."""
    try:
        subprocess.run(["dot", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def render_dot(dot_source: str, filename: str = "output", fmt: str = "png",
               output_dir: str = "output") -> str:
    """
    Render a DOT-language string to an image file using graphviz.

    Parameters
    ----------
    dot_source : str
        A valid DOT language graph description.
    filename : str
        Output filename (without extension).
    fmt : str
        Output format: 'png', 'svg', or 'pdf'.
    output_dir : str
        Directory to write the output file into.

    Returns
    -------
    str
        Path to the generated file, or empty string if graphviz is missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}.{fmt}")

    if not graphviz_available():
        # Fallback: just write the DOT source so the user can render manually
        dot_path = os.path.join(output_dir, f"{filename}.dot")
        with open(dot_path, "w") as f:
            f.write(dot_source)
        print(f"[viz] graphviz not found — DOT source saved to {dot_path}")
        print("      Install graphviz: sudo apt-get install graphviz")
        return dot_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as tmp:
        tmp.write(dot_source)
        tmp_path = tmp.name

    try:
        subprocess.run(["dot", f"-T{fmt}", tmp_path, "-o", out_path],
                        check=True, capture_output=True)
        print(f"[viz] Rendered → {out_path}")
    finally:
        os.unlink(tmp_path)
    return out_path


def _node_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


def render_tree(root, get_label, get_children, filename="tree",
                output_dir="output", fmt="png", **kw) -> str:
    """
    Generic tree renderer.  Works for ASTs, scope trees, etc.

    Parameters
    ----------
    root : Any
        The root node of the tree.
    get_label : callable(node) -> str
        Returns the display label for a node.
    get_children : callable(node) -> list
        Returns the children of a node.
    filename, output_dir, fmt :
        Forwarded to render_dot.
    **kw :
        Extra graphviz node attributes keyed by node type string (optional).
    """
    lines = ["digraph tree {", '  node [shape=box, fontname="Courier"];']
    counter = [0]

    def visit(node) -> str:
        nid = _node_id("n", counter[0])
        counter[0] += 1
        label = get_label(node).replace('"', '\\"')
        attrs = f'label="{label}"'
        # Optional per-type styling
        node_type = type(node).__name__
        if node_type in kw:
            attrs += ", " + kw[node_type]
        lines.append(f'  {nid} [{attrs}];')
        for child in get_children(node):
            cid = visit(child)
            lines.append(f"  {nid} -> {cid};")
        return nid

    visit(root)
    lines.append("}")
    return render_dot("\n".join(lines), filename, fmt, output_dir)


def render_dag(nodes, edges, filename="dag", output_dir="output", fmt="png",
               node_attrs=None, edge_labels=None, rankdir="TB") -> str:
    """
    Render a directed acyclic graph.

    Parameters
    ----------
    nodes : dict[str, str]
        Mapping from node id to display label.
    edges : list[tuple[str, str]]
        List of (src_id, dst_id) edges.
    node_attrs : dict[str, str] | None
        Optional per-node graphviz attributes.
    edge_labels : dict[tuple[str,str], str] | None
        Optional edge labels keyed by (src, dst).
    """
    lines = [
        "digraph dag {",
        f'  rankdir={rankdir};',
        '  node [shape=box, style=filled, fillcolor="#e8f4fd", fontname="Courier"];',
    ]

    for nid, label in nodes.items():
        label_esc = label.replace('"', '\\"')
        attrs = f'label="{label_esc}"'
        if node_attrs and nid in node_attrs:
            attrs += ", " + node_attrs[nid]
        lines.append(f'  {nid} [{attrs}];')

    edge_labels = edge_labels or {}
    for src, dst in edges:
        elabel = edge_labels.get((src, dst), "")
        eattr = f' [label="{elabel}"]' if elabel else ""
        lines.append(f"  {src} -> {dst}{eattr};")

    lines.append("}")
    return render_dot("\n".join(lines), filename, fmt, output_dir)


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def plot_heatmap(data, title="Memory Access Pattern", xlabel="Column",
                 ylabel="Row", filename="heatmap", output_dir="output",
                 cmap="YlOrRd"):
    """Plot a 2D numpy array as a heatmap and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.array(data), cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Heatmap → {out_path}")
    return out_path


def plot_bar_chart(labels, values, title="Benchmark", ylabel="Time (ms)",
                   filename="benchmark", output_dir="output", colors=None):
    """Plot a simple bar chart and save to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Bar chart → {out_path}")
    return out_path


def plot_comparison(before_lines, after_lines, before_title="Before",
                    after_title="After", filename="comparison",
                    output_dir="output"):
    """Render side-by-side text comparison (e.g., IR before/after optimization)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(before_lines) * 0.35)))
    for ax, lines, title in [(ax1, before_lines, before_title),
                              (ax2, after_lines, after_title)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(len(lines), 1))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        for i, line in enumerate(lines):
            ax.text(0.02, i + 0.5, line, fontsize=9, fontfamily="monospace",
                    verticalalignment="center")

    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Comparison → {out_path}")
    return out_path


def plot_timeline(events, title="Timeline", filename="timeline",
                  output_dir="output"):
    """
    Plot a memory-timeline / Gantt-like chart.

    Parameters
    ----------
    events : list[dict]
        Each dict has keys: 'name' (str), 'start' (int), 'end' (int),
        and optionally 'color' (str).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(3, len(events) * 0.5)))

    colors_cycle = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                     "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]
    for i, ev in enumerate(events):
        color = ev.get("color", colors_cycle[i % len(colors_cycle)])
        ax.barh(i, ev["end"] - ev["start"], left=ev["start"],
                height=0.6, color=color, edgecolor="black", linewidth=0.5)
        ax.text(ev["start"] + 0.5, i, ev["name"], va="center", fontsize=9)

    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([ev["name"] for ev in events])
    ax.set_xlabel("Time step")
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()

    out_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Timeline → {out_path}")
    return out_path

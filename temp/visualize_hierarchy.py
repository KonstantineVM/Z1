"""
visualize_hierarchy.py  –  v3
Fully expands each branch once; no RecursionError on cycles.
"""

import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

ROOT_SERIES = "FA086902005"
JSON_PATH   = Path("fof_formulas_extracted.json")

# ------------------------------------------------------------------ helpers --
def load_formulas(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    formulas = {}
    for tbl in data["tables"].values():
        for s in tbl["series_list"]:
            formulas[s["series_code"]] = s
    return formulas

def children(code, formulas):
    return [c["code"] for c in formulas.get(code, {}).get("derived_from", [])]

def build_tree(node, formulas, g=None, path=None, printed=None):
    """
    DFS that:
    • Adds an edge even if it closes a cycle, but
    • Expands children only the *first* time we meet a node.
    """
    g = g or nx.DiGraph()
    path = path or set()
    printed = printed or set()

    if node in path:           # we returned here via a back‑edge → cycle close
        return g

    first_time = node not in printed
    printed.add(node)

    g.add_node(node, data_type=formulas.get(node, {}).get("data_type", "Unknown"))

    path.add(node)
    for child in children(node, formulas):
        g.add_edge(node, child)
        if first_time:                       # expand only once globally
            build_tree(child, formulas, g, path, printed)
    path.remove(node)
    return g


def ascii_print(node, g, prefix="", printed=None):
    """
    Pretty‑prints the hierarchy exactly once per node.
    If a node re‑appears downstream, stop to avoid infinite recursion.
    """
    printed = printed or set()
    if node in printed:
        return                          # already shown elsewhere
    printed.add(node)

    kids = list(g.successors(node))
    for i, child in enumerate(kids):
        conn = "└── " if i == len(kids) - 1 else "├── "
        tag  = " (source)" if g.nodes[child]["data_type"] == "Data Source" else ""
        print(prefix + conn + child + tag)
        ascii_print(child, g,
                    prefix + ("    " if i == len(kids) - 1 else "│   "),
                    printed)


# ------------------------------------------------------------------- driver --
def main():
    formulas = load_formulas(JSON_PATH)
    if ROOT_SERIES not in formulas:
        raise ValueError(f"{ROOT_SERIES} not found in JSON.")

    graph = build_tree(ROOT_SERIES, formulas)
    print(ROOT_SERIES)
    ascii_print(ROOT_SERIES, graph)

    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        nx.draw(graph, pos, labels={n: n for n in graph}, arrows=False,
                node_size=400, font_size=8, node_color="#ccddee")
        plt.title(f"Hierarchy for {ROOT_SERIES}")
        plt.tight_layout(); plt.show()
    except (ImportError, nx.NetworkXException):
        print("\nGraphviz not available—ASCII tree printed only.")

if __name__ == "__main__":
    main()


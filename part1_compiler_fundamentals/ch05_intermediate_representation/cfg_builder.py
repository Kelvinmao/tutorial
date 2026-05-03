#!/usr/bin/env python3
"""
Chapter 5 — Build a Control Flow Graph (CFG) from IR instructions.

Splits IR into basic blocks and connects them with edges representing
possible control flow paths.

Usage:
    python cfg_builder.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Basic Block Identification + Control Flow Graph Construction
#
# Historical context: Frances Allen (IBM, 1970) pioneered control flow
# analysis and the concept of basic blocks. Her work earned the 2006
# Turing Award. The CFG is the foundation of nearly all compiler
# optimizations — it lets the compiler reason about which paths the
# program can take and what facts hold at each point.
#
# Problem solved: A flat IR instruction list doesn't show control flow
# structure. The optimizer needs to know: "if I'm at instruction X, which
# instructions could execute next?" The CFG provides this.
#
# ALGORITHM — Three-step construction:
#
# Step 1: Find leaders (instructions that start new basic blocks).
#   A leader is:
#   a) The very first instruction
#   b) Any LABEL instruction (it's a branch target)
#   c) Any instruction immediately *after* a JUMP or BRANCH
#      (the fallthrough point)
#   d) FUNC_BEGIN (function entry point)
#
# Step 2: Partition instructions into basic blocks.
#   Each block starts at a leader and extends to (but not including)
#   the next leader. Within a block, execution is strictly sequential
#   — no branches in, no branches out (except at the end).
#
# Step 3: Add edges between blocks.
#   - JUMP to label L → edge to block L
#   - BRANCH cond, true_label, false_label → edges to both
#   - If the last instruction is neither jump nor return, add a
#     fall-through edge to the next sequential block.
#
# The resulting CFG is a directed graph where:
#   Nodes = basic blocks (straight-line instruction sequences)
#   Edges = possible control flow transitions
#
#   IR Instructions:               Control Flow Graph:
#
#   0: x = 10                      ┌───────────────┐
#   1: LABEL cond                  │ BB0 (entry)   │  x = 10
#   2: t0 = x > 0                 │               │
#   3: BRANCH t0, body, end       └───────┬───────┘
#   4: LABEL body                         │
#   5: x = x - 1                   ┌─────┴─────────┐
#   6: JUMP cond                   │ BB1 (cond)    │  t0 = x > 0
#   7: LABEL end                   │               │  BRANCH t0
#   8: RETURN x                    └──┬───────┬───┘
#                               true│         │false
#                            ┌─────┴───┐  ┌──┴────────┐
#                            │ BB2 (body)│  │ BB3 (end)  │
#                            │ x = x - 1│  │ RETURN x   │
#                            │ JUMP cond│  └───────────┘
#                            └────┬─────┘
#                                 │  (back edge)
#                                 └─────► BB1
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))

from ir_builder import build_ir, IRInstr, Op
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


@dataclass
class BasicBlock:
    """A basic block: a straight-line sequence of instructions."""
    label: str
    instructions: list[IRInstr] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)

    def __repr__(self):
        instrs = "\n".join(f"    {i}" for i in self.instructions)
        return f"[{self.label}] → {self.successors}\n{instrs}"


class CFG:
    """Control Flow Graph: a directed graph of basic blocks."""

    def __init__(self):
        self.blocks: dict[str, BasicBlock] = {}
        self.entry: str = ""

    def add_block(self, block: BasicBlock):
        self.blocks[block.label] = block
        if not self.entry:
            self.entry = block.label

    def add_edge(self, src: str, dst: str):
        if dst not in self.blocks[src].successors:
            self.blocks[src].successors.append(dst)
        if src not in self.blocks[dst].predecessors:
            self.blocks[dst].predecessors.append(src)


def build_cfg(instructions: list[IRInstr]) -> CFG:
    """
    Build a CFG from a flat list of IR instructions.

    Algorithm:
    1. Identify leaders (first instr, targets of jumps, instrs after jumps)
    2. Group instructions into basic blocks between leaders
    3. Add edges based on jump/branch instructions
    """
    cfg = CFG()

    if not instructions:
        return cfg

    # Step 1: Find leaders (indices that start new blocks)
    leaders = {0}  # first instruction is always a leader
    label_map: dict[str, int] = {}  # label name → instruction index

    for i, instr in enumerate(instructions):
        if instr.op == Op.LABEL:
            leaders.add(i)
            label_map[instr.dst] = i
        elif instr.op in (Op.JUMP, Op.BRANCH):
            if i + 1 < len(instructions):
                leaders.add(i + 1)
        elif instr.op == Op.FUNC_BEGIN:
            leaders.add(i)
        elif instr.op == Op.FUNC_END:
            if i + 1 < len(instructions):
                leaders.add(i + 1)

    # Step 2: Create basic blocks
    sorted_leaders = sorted(leaders)
    block_counter = 0

    for idx, leader_idx in enumerate(sorted_leaders):
        end_idx = sorted_leaders[idx + 1] if idx + 1 < len(sorted_leaders) else len(instructions)
        block_instrs = instructions[leader_idx:end_idx]

        # Determine block label
        if block_instrs and block_instrs[0].op == Op.LABEL:
            label = block_instrs[0].dst
        elif block_instrs and block_instrs[0].op == Op.FUNC_BEGIN:
            label = f"func_{block_instrs[0].dst}"
        else:
            label = f"BB{block_counter}"

        block_counter += 1
        block = BasicBlock(label=label, instructions=block_instrs)
        cfg.add_block(block)

    # Step 3: Add edges
    block_labels = list(cfg.blocks.keys())
    for i, label in enumerate(block_labels):
        block = cfg.blocks[label]
        if not block.instructions:
            continue

        last = block.instructions[-1]

        if last.op == Op.JUMP:
            target = last.dst
            if target in cfg.blocks:
                cfg.add_edge(label, target)
        elif last.op == Op.BRANCH:
            true_target = last.dst
            false_target = last.src2
            if true_target in cfg.blocks:
                cfg.add_edge(label, true_target)
            if false_target in cfg.blocks:
                cfg.add_edge(label, false_target)
        elif last.op == Op.RETURN or last.op == Op.FUNC_END:
            pass  # no successor
        else:
            # Fall-through to next block
            if i + 1 < len(block_labels):
                cfg.add_edge(label, block_labels[i + 1])

    return cfg


def print_cfg(cfg: CFG):
    """Display the CFG in the terminal."""
    for label, block in cfg.blocks.items():
        instrs = "\n".join(f"  {i}" for i in block.instructions)
        preds = ", ".join(block.predecessors) if block.predecessors else "(none)"
        succs = ", ".join(block.successors) if block.successors else "(none)"
        console.print(Panel(
            f"[dim]predecessors: {preds}[/]\n{instrs}\n[dim]successors: {succs}[/]",
            title=f"[bold cyan]{label}[/]",
            border_style="cyan",
            width=50))


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let x: int = 10
let y: int = 20
let z = x + y * 2

if z > 40:
    print(z)
else:
    print(x)
"""

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    console.print("\n" + "═" * 60)
    console.print("  MiniLang CFG Builder")
    console.print("═" * 60)
    console.print(f"\n[bold]Source:[/]\n{source}")
    console.print("─" * 60)

    instructions = build_ir(source)
    cfg = build_cfg(instructions)
    print_cfg(cfg)

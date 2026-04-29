#!/usr/bin/env python3
"""
Chapter 7 — Translate custom IR into LLVM IR using llvmlite.

Shows how a real compiler backend emits LLVM IR from its own IR.

Usage:
    python llvm_emitter.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir, IRInstr, Op

try:
    from llvmlite import ir as llvm_ir
    from llvmlite import binding as llvm_binding
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False

from rich.console import Console
from rich.panel import Panel

console = Console()


def emit_llvm_ir(instructions: list[IRInstr], func_name: str = "main") -> str:
    """
    Translate custom IR instructions into LLVM IR.

    This emitter targets a simple model:
    - All variables are i64 (64-bit integers) for simplicity
    - Variables are stored in stack-allocated slots (alloca)
    - The function returns the last assigned value
    """
    if not HAS_LLVMLITE:
        return _emit_llvm_ir_text(instructions, func_name)

    # Create LLVM module and function
    module = llvm_ir.Module(name="minilang_module")
    module.triple = llvm_binding.get_default_triple()

    # Create printf declaration for print statements
    printf_ty = llvm_ir.FunctionType(llvm_ir.IntType(32),
                                      [llvm_ir.IntType(8).as_pointer()],
                                      var_arg=True)
    printf = llvm_ir.Function(module, printf_ty, name="printf")

    # Create main function: i64 main()
    func_ty = llvm_ir.FunctionType(llvm_ir.IntType(64), [])
    func = llvm_ir.Function(module, func_ty, name=func_name)
    block = func.append_basic_block(name="entry")
    builder = llvm_ir.IRBuilder(block)

    # Track variables as stack allocations
    variables: dict[str, llvm_ir.AllocaInstr] = {}
    i64 = llvm_ir.IntType(64)

    def get_or_create_var(name: str):
        if name not in variables:
            # Create alloca at function entry
            variables[name] = builder.alloca(i64, name=name)
        return variables[name]

    def load_val(name: str):
        if name in variables:
            return builder.load(variables[name], name=f"{name}.val")
        # Try parsing as constant
        try:
            val = int(float(name))
            return llvm_ir.Constant(i64, val)
        except (ValueError, TypeError):
            # Unknown — create a zero
            return llvm_ir.Constant(i64, 0)

    last_result = None
    # Format string for printf
    fmt_str = None

    for instr in instructions:
        if instr.op == Op.LOAD_CONST:
            ptr = get_or_create_var(instr.dst)
            try:
                val = int(float(instr.src1))
            except (ValueError, TypeError):
                val = 0
            builder.store(llvm_ir.Constant(i64, val), ptr)
            last_result = instr.dst

        elif instr.op == Op.COPY:
            src = load_val(instr.src1)
            ptr = get_or_create_var(instr.dst)
            builder.store(src, ptr)
            last_result = instr.dst

        elif instr.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD):
            left = load_val(instr.src1)
            right = load_val(instr.src2)
            if instr.op == Op.ADD:
                result = builder.add(left, right, name=instr.dst)
            elif instr.op == Op.SUB:
                result = builder.sub(left, right, name=instr.dst)
            elif instr.op == Op.MUL:
                result = builder.mul(left, right, name=instr.dst)
            elif instr.op == Op.DIV:
                result = builder.sdiv(left, right, name=instr.dst)
            elif instr.op == Op.MOD:
                result = builder.srem(left, right, name=instr.dst)
            ptr = get_or_create_var(instr.dst)
            builder.store(result, ptr)
            last_result = instr.dst

        elif instr.op in (Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE):
            left = load_val(instr.src1)
            right = load_val(instr.src2)
            cmp_map = {Op.EQ: "==", Op.NE: "!=", Op.LT: "<",
                       Op.GT: ">", Op.LE: "<=", Op.GE: ">="}
            cmp = builder.icmp_signed(cmp_map[instr.op], left, right)
            result = builder.zext(cmp, i64, name=instr.dst)
            ptr = get_or_create_var(instr.dst)
            builder.store(result, ptr)

        elif instr.op == Op.NEG:
            val = load_val(instr.src1)
            result = builder.neg(val, name=instr.dst)
            ptr = get_or_create_var(instr.dst)
            builder.store(result, ptr)

        elif instr.op == Op.PRINT:
            if fmt_str is None:
                fmt = "%ld\n\0"
                c_fmt = llvm_ir.Constant(llvm_ir.ArrayType(llvm_ir.IntType(8), len(fmt)),
                                          bytearray(fmt.encode("utf8")))
                fmt_global = llvm_ir.GlobalVariable(module, c_fmt.type, name="fmt")
                fmt_global.linkage = "internal"
                fmt_global.global_constant = True
                fmt_global.initializer = c_fmt
                fmt_str = fmt_global

            val = load_val(instr.src1)
            fmt_ptr = builder.bitcast(fmt_str,
                                       llvm_ir.IntType(8).as_pointer())
            builder.call(printf, [fmt_ptr, val])

        elif instr.op == Op.LABEL:
            new_block = func.append_basic_block(name=instr.dst)
            if not builder.block.is_terminated:
                builder.branch(new_block)
            builder = llvm_ir.IRBuilder(new_block)

        elif instr.op == Op.JUMP:
            target_block = None
            for b in func.blocks:
                if b.name == instr.dst:
                    target_block = b
                    break
            if target_block and not builder.block.is_terminated:
                builder.branch(target_block)

        elif instr.op == Op.BRANCH:
            cond = load_val(instr.src1)
            cond_bool = builder.icmp_signed("!=", cond,
                                             llvm_ir.Constant(i64, 0))
            true_block = false_block = None
            for b in func.blocks:
                if b.name == instr.dst:
                    true_block = b
                if b.name == instr.src2:
                    false_block = b
            if true_block is None:
                true_block = func.append_basic_block(name=instr.dst)
            if false_block is None:
                false_block = func.append_basic_block(name=instr.src2)
            if not builder.block.is_terminated:
                builder.cbranch(cond_bool, true_block, false_block)
            builder = llvm_ir.IRBuilder(true_block)

        # Skip func_begin, func_end, param, return for simple demo

    # Return the last computed value (or 0)
    if last_result:
        ret_val = load_val(last_result)
    else:
        ret_val = llvm_ir.Constant(i64, 0)

    if not builder.block.is_terminated:
        builder.ret(ret_val)

    # Ensure all blocks are terminated
    for block in func.blocks:
        if not block.is_terminated:
            llvm_ir.IRBuilder(block).ret(llvm_ir.Constant(i64, 0))

    return str(module)


def _emit_llvm_ir_text(instructions: list[IRInstr], func_name: str) -> str:
    """Fallback: generate LLVM IR as plain text when llvmlite is not available."""
    lines = [
        f'; LLVM IR generated from MiniLang (text fallback)',
        f'define i64 @{func_name}() {{',
        'entry:',
    ]

    for instr in instructions:
        if instr.op == Op.LOAD_CONST:
            try:
                val = int(float(instr.src1))
            except (ValueError, TypeError):
                val = 0
            lines.append(f'  %{instr.dst} = add i64 {val}, 0')
        elif instr.op == Op.COPY:
            lines.append(f'  %{instr.dst} = add i64 %{instr.src1}, 0')
        elif instr.op == Op.ADD:
            lines.append(f'  %{instr.dst} = add i64 %{instr.src1}, %{instr.src2}')
        elif instr.op == Op.SUB:
            lines.append(f'  %{instr.dst} = sub i64 %{instr.src1}, %{instr.src2}')
        elif instr.op == Op.MUL:
            lines.append(f'  %{instr.dst} = mul i64 %{instr.src1}, %{instr.src2}')

    lines.append('  ret i64 0')
    lines.append('}')
    return '\n'.join(lines)


# ── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE = """\
let a = 2 + 3
let b = a * 4
let c = b - 10
"""

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE

    console.print("\n[bold]═══ LLVM IR Emitter ═══[/]\n")
    console.print(f"[bold]Source:[/]\n{source}")

    ir = build_ir(source)
    console.print("[bold]Custom IR:[/]")
    for i in ir:
        console.print(f"  {i}")

    console.print("\n[bold]Generated LLVM IR:[/]")
    llvm_text = emit_llvm_ir(ir)
    console.print(Panel(llvm_text, title="[cyan]LLVM IR[/]", border_style="cyan"))

    if not HAS_LLVMLITE:
        console.print("[yellow]Note: llvmlite not installed — showing text fallback.[/]")
        console.print("[yellow]Install with: pip install llvmlite[/]")

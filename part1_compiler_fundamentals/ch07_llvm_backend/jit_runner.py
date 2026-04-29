#!/usr/bin/env python3
"""
Chapter 7 — JIT compile and execute LLVM IR.

Compiles LLVM IR to native code and runs it in-process — no external
compiler needed!

Usage:
    python jit_runner.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch02_lexer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_parser_ast"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_intermediate_representation"))

from ir_builder import build_ir
from llvm_emitter import emit_llvm_ir
from llvm_optimizer import optimize_llvm_ir

from rich.console import Console

console = Console()

try:
    from llvmlite import binding as llvm
    import ctypes
    HAS_LLVMLITE = True
except ImportError:
    HAS_LLVMLITE = False


def jit_execute(llvm_ir_str: str, func_name: str = "main",
                optimize: bool = True) -> int:
    """
    JIT-compile LLVM IR and execute a function, returning its return value.
    """
    if not HAS_LLVMLITE:
        raise RuntimeError("llvmlite required for JIT execution")

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Optionally optimize
    if optimize:
        llvm_ir_str = optimize_llvm_ir(llvm_ir_str, opt_level=2)

    # Parse and verify
    mod = llvm.parse_assembly(llvm_ir_str)
    mod.verify()

    # Create execution engine
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    # Add our module
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()

    # Get function pointer and call it
    func_ptr = engine.get_function_address(func_name)
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int64)(func_ptr)
    result = cfunc()

    return result


SAMPLE = """\
let a = 7
let b = 8
let c = a * b
"""

if __name__ == "__main__":
    if not HAS_LLVMLITE:
        console.print("[red]llvmlite is required for JIT execution.[/]")
        console.print("[yellow]Install with: pip install llvmlite[/]")
        sys.exit(0)

    source = sys.argv[1] if len(sys.argv) > 1 else SAMPLE

    console.print("\n[bold]═══ LLVM JIT Runner ═══[/]\n")
    console.print(f"[bold]Source:[/]\n{source}")

    ir = build_ir(source)
    llvm_ir_str = emit_llvm_ir(ir)

    console.print(f"\n[dim]JIT compiling and executing...[/]")
    result = jit_execute(llvm_ir_str, optimize=True)

    console.print(f"\n[bold green]Result: {result}[/]")
    console.print(f"[dim](This was computed by native machine code "
                  f"generated from LLVM IR!)[/]")

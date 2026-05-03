#!/usr/bin/env python3
"""
Chapter 7 — JIT compile and execute LLVM IR.

Compiles LLVM IR to native code and runs it in-process — no external
compiler needed!

Usage:
    python jit_runner.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Just-In-Time (JIT) Compilation via LLVM MCJIT
#
# Historical context: JIT compilation was pioneered by Smalltalk-80 (1983)
# and later by Java's HotSpot (1999). The idea: instead of writing object
# files to disk and linking them (ahead-of-time compilation), compile
# directly to executable machine code in memory and call it immediately.
# LLVM's MCJIT engine provides this capability for any language that can
# emit LLVM IR.
#
# Problem solved: For interactive use (REPLs, notebooks, auto-tuning),
# we want to compile and run code instantly without an external compiler
# toolchain. JIT also enables runtime specialization — compiling code
# with runtime values baked in as constants.
#
# How it works:
# 1. Initialize LLVM's native target (x86_64, ARM, etc.).
# 2. Optionally optimize the LLVM IR (using llvm_optimizer.py).
# 3. Parse and verify the LLVM IR module.
# 4. Create a TargetMachine for the host CPU.
# 5. Create an MCJIT ExecutionEngine (this allocates executable memory).
# 6. Add the module to the engine and finalize (triggers code generation
#    — LLVM performs instruction selection, register allocation,
#    scheduling, and binary encoding).
# 7. Get the function pointer address via get_function_address().
# 8. Use ctypes to create a callable Python function from the pointer.
# 9. Call it — we're executing native machine code generated from our
#    high-level source!
#
#   Source code        LLVM IR            Machine Code       Result
#   ┌──────────┐    ┌─────────────┐   ┌────────────┐   ┌──────┐
#   │ y = 4*x+1│─►│ mul, add,   │─►│ 48 89 f8   │─►│  21  │
#   │          │   │ ret         │   │ 48 c1 e0.. │   │      │
#   └──────────┘   └─────────────┘   │ 48 83 c0 01│   └──────┘
#     ch01-06         ch07              │ c3         │  (x=5)
#     (frontend)      (emit)            └────────────┘
#                                        MCJIT engine
#                                        (in-memory,
#                                         no .o file)
#
#   Python side:  fn = ctypes.CFUNCTYPE(c_int64, c_int64)(addr)
#                 result = fn(5)  # calls native code!
#
# This is exactly how Julia, Numba, and AI compilers execute generated code.
# ═══════════════════════════════════════════════════════════════════════════

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

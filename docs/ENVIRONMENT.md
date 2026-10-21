# SignalDNA - Environment Plan

Status: M0 proposal

Platform: Windows 11.
Python: standalone CPython 3.12 or 3.13, pending installation verification.
Virtual environment: C:\Projects\SignalDNA\.venv.
Dependencies: NumPy and Matplotlib initially.
Dependency versions: pin after environment validation.
Hardware: CPU-first; GPU acceleration not required.

## Isolation
Do not modify Espressif-managed Python.
Do not install project dependencies globally.
Do not modify system PATH without explicit approval.

## Current Discovery
Python launcher unavailable; common locations and registry checks found no interpreter.
Environment installation deferred until M0 approval.

## Verified M1 Environment
Python: 3.13.15
NumPy: 2.5.4
Matplotlib: 3.11.2
Virtual environment: C:\Projects\SignalDNA\.venv
Dependency snapshot: requirements.lock.txt
Import verification: PASS (exit 0).

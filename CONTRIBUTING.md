# Contributing Guide

This project uses Poetry and a `src/` layout. Please follow these rules to keep the codebase consistent and maintainable.

## Environment
- Python: 3.10 (managed by pyenv via `.python-version`)
- Package manager: Poetry
- Install: `poetry install`
- Run tests: `poetry run pytest -q`

## Code Style
- Docstrings: English only. Keep summaries concise and informative.
- Type hints: Use for all public functions and classes (parameters and return types).
- Naming: Descriptive, avoid abbreviations. No 1–2 character variable names.
- Control flow: Prefer early returns over deep nesting. Raise meaningful exceptions only.
- Comments: Only for non-obvious rationale, invariants, or edge cases.

## Project Structure
- `src/torch_camera_design/`: library code
  - `losses/`: loss functions (`luther.py`, `vora.py`, `l2.py`)
  - `evaluation/`: metrics and reports
- `tests/`: unit tests

## Public API
- Keep `__init__.py` light. Re-export only frequently used symbols.
- Avoid heavy imports at package import time.
- Preserve backward compatibility when reasonable. Breaking changes need a version bump and notes.

## Docstrings Convention
- Use NumPy-style or Google-style sections (Parameters, Returns, Raises, Notes).
- Include shapes/dtypes for tensors and arrays when relevant.
- Example (NumPy style):
  ```python
  def foo(x: torch.Tensor) -> torch.Tensor:
      """Compute something.

      Parameters
      ----------
      x : torch.Tensor, shape (N, C)
          Input tensor.

      Returns
      -------
      torch.Tensor
          Output tensor with the same shape as ``x``.
      """
  ```

## Commits & PRs
- Message format: `type(scope): description`, e.g. `feat(losses): add vora loss`.
- Keep commits focused; avoid mixing unrelated changes.
- Include tests for new features and bug fixes when applicable.

## Testing
- Cover edge cases (shape mismatches, degenerate inputs, dtypes/devices).
- Keep tests deterministic.

## Releasing
- Version is defined in `pyproject.toml` and mirrored in `src/torch_camera_design/_version.py`.
- Follow semantic versioning.

Thank you for contributing!

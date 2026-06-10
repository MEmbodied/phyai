# Repository Guidelines

Read CLAUDE.md as AGENTS.md!
Read CLAUDE.md as AGENTS.md!
Read CLAUDE.md as AGENTS.md!
Read CLAUDE.md as AGENTS.md!
Read CLAUDE.md as AGENTS.md!
Read CLAUDE.md as AGENTS.md!

Follow the instructions in CLAUDE.md!

Read skills from .claude directory!
Read skills from .claude directory!
Read skills from .claude directory!
Read skills from .claude directory!
Read skills from .claude directory!
Read skills from .claude directory!

## Project Structure & Module Organization
This is a `uv` workspace monorepo for Python 3.12+. The main library lives in `phyai/src/phyai`, with tests in `phyai/tests`. `phyai-kernel/` contains Triton and JIT kernel code under `phyai_kernel/`, plus `tests/` and `benchmark/`. `phyai-ext/` contains the C++ extension: `csrc/`, `include/`, Python wrappers in `src/phyai_ext`, and tests in `tests/`. `phyai-model-optimizer/` and `phyai-utils-tools/` follow the same `src/<package>` and `tests/` layout. Documentation is in `docs/`, examples in `examples/`, helper scripts in `scripts/`, and Dockerfiles in `docker/`.

## Build, Test, and Development Commands
- `uv sync`: install the full editable workspace from `uv.lock`; this also builds `phyai-ext` via scikit-build-core.
- `uv run pytest`: run the full test suite defined in the root `pyproject.toml`.
- `uv run pytest phyai/tests/weights/test_loader.py`: run one test module; append `::test_name` for a single test.
- `scripts/setup_dev_env.sh`: install pre-commit hooks for local development.
- `scripts/run_pre_commit.sh`: run formatting, spelling, and C/C++ formatting checks across the repo.
- `uv run phyai-kernel show-env`: inspect kernel-related environment and build settings.

## Coding Style & Naming Conventions
Python is formatted with `ruff-format`; keep modules typed where practical and use package-local helpers before adding new abstractions. C and C++ use `.clang-format` with 2-space indentation and 128-column formatting. `.editorconfig` requires LF endings, UTF-8, and final newlines. C++ naming is enforced through `.clang-tidy`: classes use `CamelCase`, variables use `lower_case`, and globals use `UPPER_CASE`.

## Testing Guidelines
Tests use `pytest` with importlib import mode. Place tests in the package-local `tests/` tree and name files `test_*.py`. Prefer targeted tests for changed behavior, then run `uv run pytest` before submitting broader changes. CPU is the default for most tests; CUDA-specific tests should opt in explicitly through device selection.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit-style messages such as `feat(phyai): pi0.5 support` and `fix(phyai): weight loader refactor`. Use a concise type and scope when possible: `feat(phyai-kernel): add rms_norm benchmark`. Pull requests should describe the change, list validation commands run, link related issues, and include screenshots or logs when user-visible behavior, docs, or performance output changes.

## Security & Configuration Tips
Keep dependency pins and `uv.lock` changes intentional. Do not casually bump hard runtime dependencies such as Torch, FlashInfer, or Transformers. Declare new `PHYAI_*` environment variables in `phyai/src/phyai/env.py` instead of reading `os.environ` ad hoc.

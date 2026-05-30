---
name: paper-pdf-ingest
description: Use this skill when the user provides only a PDF paper and asks Codex to understand, implement, debug, or design experiments from it. Convert the PDF into Codex-readable verified specs. Do not trust raw PDF-to-Markdown math blindly.
---

# Paper PDF Ingest Skill

## Purpose

Convert an academic PDF into reliable Codex-readable specifications. This skill is mandatory when the paper source is only PDF and no LaTeX source is available.

## Required Workflow

1. Never directly implement algorithms from raw PDF text.
2. Convert the PDF using at least one strong academic PDF parser:
   - Prefer MinerU for scientific PDFs with formulas, tables, multi-column layouts, and OCR needs.
   - Use Marker as a second parser or cross-check.
   - Use MarkItDown only as a lightweight fallback for ordinary documents.
3. Save parser outputs under `docs_for_codex/parsed/`.
4. Save verified specs under `docs_for_codex/verified/`.
5. Create these verified files:
   - `00_notation_table.md`
   - `01_system_model.md`
   - `02_tensor_model.md`
   - `03_algorithm_steps.md`
   - `04_projection_rules.md`
   - `05_experiment_protocol.md`
   - `06_known_failure_modes.md`
6. Mark uncertain formulas with `[FORMULA_CHECK_REQUIRED]`.
7. For matrix/tensor algorithms, explicitly extract:
   - dimensions
   - conjugate transpose conventions
   - Khatri-Rao / Kronecker / Hadamard order
   - normalization and scale absorption rules
   - objective functions and acceptance criteria
8. Do not modify code until the relevant verified spec file exists.

## Commands

Try Marker:

```bash
conda run -n pdf-marker marker_single <paper.pdf> \
  --output_dir docs_for_codex/parsed/marker \
  --output_format markdown \
  --force_ocr
```

Try MinerU:

```bash
conda run -n pdf-mineru mineru \
  -p <paper.pdf> \
  -o docs_for_codex/parsed/mineru \
  -b pipeline
```

## Quality Checks

Before using the parsed paper for code:
- Compare formula-heavy sections between Marker and MinerU output.
- Check algorithm pseudocode against surrounding text.
- Check that every variable in the algorithm has a dimension.
- Check that all equations referenced by code are copied into verified specs.
- If formulas disagree across tools, ask the user or mark them unresolved.

## Forbidden Behavior

- Do not treat PDF-converted equations as ground truth without verification.
- Do not implement CRB, FIM, VP-WNLS, CPD, or projection code from OCR text alone.
- Do not silently repair missing mathematical symbols.

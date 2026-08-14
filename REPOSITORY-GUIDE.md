# Repository Guide

This repository contains several connected lines of work developed during the same course project. The current structure keeps the geometric-autoencoder project central while preserving the earlier style-transfer, CNN, geometry, and topology studies in their original locations.

## Directory Map

| Path | Role | Contents | Kept in place because |
| --- | --- | --- | --- |
| `entering-las-meninas-10-minute-presentation.pptx` | Final presentation | Ten-slide project overview with English speaker notes | It is the single canonical presentation at the repository root. |
| `entering-las-meninas-10-minute-presentation/` | Presentation source | PPTD source, editable pages, media, QA renders, notes, and archived exports | The manifest and pages use internal relative paths. |
| `OH_Programming/` | Core experiment | Geometric autoencoder, model checkpoint, outputs, and submission notebooks | Scripts and notebooks resolve assets relative to this directory. |
| `outputs/artwork/` | Main project media | Final static artwork and documentation | Several project documents reference these paths. |
| `outputs/video/` | Main project media | Final parallax animation and documentation | The extended pipeline validates outputs at this path. |
| `style_transfer/` | Supporting experiment | Self-contained VGG-19 style-transfer study | The package can be run independently. |
| `EXTENDED-CNN-STUDY.md` | Extended study | Earlier VGG, geometry, topology, and mirror-ablation narrative | Root-relative links remain valid. |
| `src/` | Extended-study code | Feature extraction, geometry analysis, transformations, aggregation, and visualization | The existing pipeline imports these modules. |
| `scripts/` | Project tooling | Downloads, pipeline execution, validation, and public-release checks | Scripts assume the existing output layout. |
| `tests/` | Automated validation | Unit, artifact, and release-safety tests | Tests check fixed paths and expected outputs. |
| `report/` | Extended writing | Long-form report, sources, and Chinese translation | Preserved as project history. |
| `outputs/figures/`, `outputs/metrics/` | Extended results | Quantitative figures and tables | Consumed by reports and validation scripts. |
| `presentation/`, `outputs/presentation/` | Earlier presentation work | Extended-study deck builder and generated presentation | The builder exports to this fixed location. |
| `assignment-1-first-delivery-zh.md` | Earlier course work | Chinese draft of the first delivery | Preserved as project history. |
| `assignment-1-submission/` | Local course archive | Submission package containing materials unsuitable for public redistribution | Intentionally excluded by `.gitignore`. |
| `pics/` | Local research material | Large source images with mixed redistribution status | Intentionally excluded by `.gitignore`. |

## Path Stability

The repository was reorganized through documentation and navigation rather than by moving implementation directories. Several paths have real runtime dependencies:

- `OH_Programming/autoencoder-experiment.py` resolves the artwork, inputs, outputs, and model relative to its own directory.
- `OH_Programming/build-submission-notebooks.py` and both notebooks locate files under `OH_Programming/submission/`.
- The extended root pipeline writes to `outputs/`, where tests and validation scripts expect fixed artifacts.
- `presentation/build-deck.mjs` exports to `outputs/presentation/`.
- The PPTD manifest, editable page files, and presentation media use paths internal to the presentation source directory.

Keeping these directories stable avoids unnecessary regressions while the root README provides a clear account of how the pieces relate.

## Presentation Versions

The canonical presentation is `entering-las-meninas-10-minute-presentation.pptx` at the repository root. Additional exports remain inside the presentation source directory for reproducibility and version history; they are not separate project entry points.

## Local, Unpublished Material

The ignored `.cache/`, `.venv/`, `pics/`, and `assignment-1-submission/` directories remain local. They are not uncommitted Git changes. Caches and environments are reproducible, while the research images and course archives remain excluded because of size or redistribution constraints.

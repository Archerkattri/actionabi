# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-09-17

### Fixed
- Version-pinned CLI contract tests (`cli_version`, `install_smoke`) track the
  release version; v1.1.0's bump commit landed the pins one commit late.

## [1.1.0] - 2026-09-17

### Added

- Installed-artifact packaging with packaged-build tests.
- Link to the Hugging Face 35-dataset action-convention audit dataset.
- Live GitHub downloads badge and Zenodo DOI
  (10.5281/zenodo.21500715) in README/CITATION.
- This changelog and the third-party NOTICE file.

### Changed

- Refreshed README visuals in publication style; dropped the CI badge per
  badge policy; updated CI to Node 24 runtimes.

## [1.0.0] - 2026-07-22

Forensic action-interface recovery with calibrated abstention: contract and
trajectory parsing, CPU evidence scoring, evidence records, identifiability
analysis, active probing, and convention conversion, with a CLI, a Catch2
test suite, an optional CUDA parity path and pybind scoring module, an
optional pinocchio dynamics backend, model adapters, and paper figures.

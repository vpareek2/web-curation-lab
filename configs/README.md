# Configuration

Versioned configuration belongs in these directories:

- `models/`: architecture definitions.
- `training/`: optimizer, schedule, precision, and distributed settings.
- `data/`: corpus stages, mixtures, tokenization, and sequence packing.
- `evaluation/`: benchmark suites and reporting settings.

Every full training run should be reconstructible from committed configuration
plus immutable input manifest identifiers.

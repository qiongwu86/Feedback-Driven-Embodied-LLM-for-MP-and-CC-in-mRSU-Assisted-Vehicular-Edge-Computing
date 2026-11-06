# FD-EMC

Source code accompanying the following paper:

**Feedback-Driven Embodied LLM for Coupled Motion Planning and Cooperative Caching in mRSU-Assisted Vehicular Edge Computing**

## Installation

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

The `datasets/` directory contains the preprocessed MovieLens-1M data used by this code:

- `ml1m_train.pkl`: training records read by the simulator.
- `ml1m_test.pkl`: test records read by the simulator.
- `ml1m_train.csv` and `ml1m_test.csv`: corresponding tabular files.

These are the project's preprocessed subsets, not the complete original MovieLens-1M release. Data use is subject to the original MovieLens terms in `datasets/MovieLens-1M.README`.

# Contributing

Thanks for your interest in improving FraudShield AI. This is an educational project; please keep all examples synthetic and never submit private customer or payment data.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements-dev.txt
```

## Before opening a pull request

```bash
pytest -q
python -m compileall -q src scripts app.py
```

Please include tests for behavior changes, document assumptions, and avoid reporting synthetic results as real-world performance.

# Mixtum: The geometry of admixture in population genetics

## Development notes
    python -m venv .venv
    source .venv/bin/activate
    pip install panel watchfiles matplotlib
    panel convert mixtum-gui.py --to pyodide-worker --out docs
    mv docs/mixtum-gui.html docs/index.html

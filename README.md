# Mixtum: The geometry of admixture in population genetics

## Development notes
    python -m venv .venv
    source .venv/bin/activate
    pip install panel watchfiles matplotlib
    panel convert mixtum-gui.py --to pyodide-worker --out gui
    mv gui/mixtum-gui.html gui/index.html

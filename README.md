# Mixtum: The geometry of admixture in population genetics

## Overview

Mixtum has been developed as a standalone Python script and as a graphical user interface. Moreover, an experimental website version exists which can be visited [here](https://jmcastelo.github.io/mixtum/). Before running Mixtum, please ensure the required dependencies are met. Instructions to install them on a Linux system are given below.

### Standalone script

To be run from the command line, this script needs some arguments which inform it where the needed input files are located, where to save the resulting output files and the number of parallel processes with which to perform the computations.

To get help, please run the following command:

    python mixtum.py --help

### Graphical user interface

A GUI has been developed with Qt's `PySide6` library. Please run it with the following command:

    python mixtum_gui.py

## Dependencies

We give instructions to install the dependencies for a Linux system using a virtual environment and `pip`.

First create a virtual environment under the `.venv` directory, where the required libraries will be installed, and activate it.

    python -m venv .venv
    source .venv/bin/activate

The standalone script depends on `numpy` and `matplotlib`. The graphical user interface depends on `pyside6`, `numpy` and `matplotlib`. Please, install these libraries in the virtual environment with `pip` as follows.

    pip install pyside6 numpy matplotlib

## Usage instructions

To do...

## Development notes

These notes are meant to be read by the developers of Mixtum.

The experimental website version of the GUI can be set up for development as follows. If it does not exist, first create a virtual environment, activate it and install `panel`, `watchfiles` and `matplotlib`.

    python -m venv .venv
    source .venv/bin/activate
    pip install panel watchfiles matplotlib
    
To start delevopment run the following command which will open a web browser to watch the changes on the code as they are made.

    panel serve mixtum_panel.py --show --autoreload

To convert the code into a website, run the following commands.

    panel convert mixtum_panel.py --to pyodide-worker --out docs
    mv docs/mixtum_panel.html docs/index.html

The website will be available as a GitHub page on the repository.

Note that in order for some Panel features to work we have used a beta version installed from the GitHub repository of Panel, and converted Mixtum to `pyscript` format. Once these features have been incorporated into the main Panel release, one can convert to `pyodide-worker` format as above.

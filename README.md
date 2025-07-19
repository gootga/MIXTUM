# Mixtum: a graphical tool for two-way admixture analysis in population genetics based on $f$-statistics

## Abstract

Mixtum is a Python-based code that provides estimates of ancestry contributions in a hybrid derived from a two-way admixture based on bi-allelic genotype data.
The outcomes of Mixtum are based on the geometric interpretation of the  $f$-statistics formalism. Designed with the intent of being user-friendly as a priority, Mixtum allows to handle interactively a menu of populations provided by the user to build up different admixture models altogether with the set of auxiliary populations required by the framework. The results are presented graphically. More importantly, Mixtum provides a novel index (an angle) that assesses the quality of the ancestral reconstruction of the model under scrutiny. The conventional quantities such as the  $f_3$ admixture test and the $f_4$-ratio statistics are also provided.

### References

* J.-A. Oteo and G. Oteo-García. The geometry of admixture in population genetics: the blessing of dimensionality. Genetics, 228(2):iyae134, 08 2024. ISSN 1943-2631. doi: 10.1093/genetics/iyae134. URL https://doi.org/10.1093/genetics/iyae134
* G. Oteo-García and J.-A. Oteo. A Geometrical Framework for f-Statistics. Bulletin of Mathematical Biology, 83:14, 2021. ISSN 1522-9602. doi: 10.1007/s11538-020-00850-8. URL https://doi.org/10.1007/s11538-020-00850-8

## Overview

Mixtum has been developed as a standalone Python script and as a graphical user interface (GUI). The GUI has been more thoroughly tested than the script version. Before running Mixtum, please ensure the required dependencies are met. Instructions to install them on Linux and Windows systems are given below. Alternatively, download the release suitable for your operating system, which contains a standalone executable. In this case, the dependencies are included in the executable and don't need to be installed.

### Standalone script

To be run from the command line, this script needs some arguments which inform it where the needed input files are located, where to save the resulting output files and the number of parallel processes with which to perform the computations.

To get help, please run the following command:

    python mixtum.py --help

### Graphical user interface

A GUI has been developed with Qt's `PySide6` library. Please, download it and run the executable, or get the source code and run it with the following command:

    python mixtum_gui.py

## Dependencies

We give instructions to install the dependencies for Linux and Windows systems using a virtual environment and `pip`. We assume that Python 3 is available on your system.

First create a virtual environment under the `.venv` directory within Mixtum's location, where the required libraries will be installed, and activate it. Note that you can create the virtual environment on any location of your filesystem, and name it as you wish.

On Linux and Windows, open a terminal and enter Mixtum's directory. Then create the virtual environment:

    python -m venv .venv

Afterwards, you should activate it. On Linux, this is done as follows:

    source .venv/bin/activate

On Windows, the activation script to be run depends on the employed shell. If using Command Prompt:

    .venv\Scripts\activate.bat

Whereas PowerShell requires running the following script:

    .venv\Scripts\Activate.ps1

In case the execution policiy of PowerShell does not allow the execution of scripts, you can enable it with the following command on a PowerShell run as administrator:

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

The standalone script depends on `numpy` and `matplotlib`. The graphical user interface depends on `pyside6`, `numpy` and `matplotlib`. Please, install these libraries in the virtual environment with `pip` as follows.

    pip install pyside6 numpy matplotlib

## Usage instructions

### Standalone script

### Graphical user interface

## Building the executable

We make use of [pyinstaller](https://pyinstaller.org/en/stable/) to build self-contained standalone executables for several operating systems. Note that the executable for each operating system must be built under that particular operating system, no cross-compiling is possible with this tool.

First install `pyinstaller` in the virtual environment that contains the dependencies of Mixtum.

    pip install pyinstaller

Then, if under Linux run it in the project's root directory as follows.

    pyinstaller --onefile mixtum_gui.py

Or if under Windows,

    pyinstaller --onefile --windowed mixtum_gui.py

to avoid providing a console window.

If everything works fine, the executable can be found in the `dist` subdirectory.

## Website development notes

These notes are meant to be read by the developers of Mixtum.

The experimental website version of the GUI can be set up for development as follows. First install `panel` and `watchfiles`, in the virtual environment that contains the rest of the dependencies.

    pip install panel watchfiles
    
The website uses [Panel](https://panel.holoviz.org/) framework for the dashboard design. The code of the dashboard and that of the computation functions, is contained in a single Python script named `mixtum_panel.py`, without other additional scripts needed.

To start delevopment run the following command which will open a web browser to watch the changes on the code as they are made.

    panel serve mixtum_panel.py --show --autoreload

To convert the code into a website, run the following commands.

    panel convert mixtum_panel.py --to pyodide-worker --out docs
    mv docs/mixtum_panel.html docs/index.html

The website will be available as a GitHub page on the repository, once it has been activated.

Note that in order for some Panel features to work we have used a beta version installed from the GitHub repository of Panel, and converted Mixtum to `pyscript` format. Once these features have been incorporated into the main Panel release, one can convert to `pyodide-worker` format as above.

# Mixtum: a graphical tool for two-way admixture analysis in population genetics based on $f$-statistics

## Abstract

Mixtum is a Python-based code that estimates ancestry contributions in a hybrid derived from a two-way admixture based on bi-allelic genotype data. The outcomes of Mixtum come from the geometric interpretation of the  $f$-statistics formalism. Designed with user-friendliness as a priority, Mixtum allows to interactively handle a menu of user-supplied populations to build different mixture models in conjunction with the set of auxiliary populations required by the framework. The results are presented graphically, including principal components plots of the allele frequencies in the dataset. More importantly, Mixtum provides a novel index (an angle) that assesses the quality of the ancestral reconstruction of the model under scrutiny. The conventional statistics $f_2$, $f_3$, $f_4$ as well as the  $f_3$ admixture test and the $f_4$-ratio are also provided.

<img width="615" height="454" alt="image" src="https://github.com/user-attachments/assets/bb2b5da0-1ba1-43e2-bc39-78ab4f7002f0" />


### References

* J.-A. Oteo and G. Oteo-García. The geometry of admixture in population genetics: the blessing of dimensionality. Genetics, 228(2):iyae134, 08 2024. ISSN 1943-2631. doi: 10.1093/genetics/iyae134. URL https://doi.org/10.1093/genetics/iyae134
* G. Oteo-García and J.-A. Oteo. A Geometrical Framework for f-Statistics. Bulletin of Mathematical Biology, 83:14, 2021. ISSN 1522-9602. doi: 10.1007/s11538-020-00850-8. URL https://doi.org/10.1007/s11538-020-00850-8



## Overview

Mixtum has been developed as a standalone Python script and as a graphical user interface (GUI). The GUI has been more thoroughly tested than the script version, so use the script at your own risk. Before running Mixtum, please ensure the required dependencies are met. Instructions to install them on Linux and Windows systems are given below. Alternatively, download the release suitable for your operating system, which contains a standalone executable. In this case, the dependencies are included in the executable and don't need to be installed.

### Graphical user interface

A GUI has been developed with Qt's `PySide6` library. Please, download the release and run the executable, or get the source code and run it with the following command:

    python mixtum_gui.py

### Standalone script

To be run from the command line, this script needs some arguments which inform it where the needed input files are located, where to save the resulting output files and the number of parallel processes with which to perform the computations. Note that the script version of Mixtum is not as complete and well-tested than the GUI version.

To get help, please run the following command:

    python mixtum.py --help

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

## Usage instructions (GUI)

Mixtum is organized as a set of tabs, each of which contains a step of the workflow. Below these tabs, a log console displays information relevant to each step. The workflow may be described as follows. 

#### 1. Input files.

It is compulsory to select the triad '*.geno, *.ind, *.snp'. They are in EIGENSTRAT format. Optionally, you can select and load a plain text file with a list of populations of interest, in a column. Then, parse and check the input files. This may take several minutes depending on the size of the dataset and the CPU.

#### 2. Populations

The left table contains all the population names in '*.ind'. They may be ordered alphabetically by clicking on the table's header, and selected/deselected with the mouse left button. Note that clicking while pressing the `shift` key allows you to select a range of populations, or pick specific ones with the `ctrl` key. The combination `ctrl + a` allows the selection of all populations. 

After searching and choosing populations of interest, select the number of computation processes to parallelize the frequencies computation. Check how many cores your CPU has to tune this parameter. Then compute the frequencies to build up the table of allele frequencies on which all the f-statistics are computed or, equivalently, on which all the scalar products are carried out.

#### 4. Admixture model
   
Choose the admixture model and the set of auxiliary populations (at least 3) on the left side of the tab. Then compute the results. Ensure a reasonable stability of the outcomes with respect to the cardinal of the set. You can then examine the linear regression plot of the admixture model, and identify the auxiliary pairs by clicking the corresponding plot points. A glimpse may already inform you about the quality of the admixture model. Examine the angles, too. A good ancestral admixture reconstruction conveys the Post-JL angle is close to 180 degrees. A decreasing of the angle from Pre-JL to Post-JL points towards a bad admixture model.

#### 5. PCA

Choose populations of interest to visualize their allele frequencies in terms of the Principal Components content in 2D and 3D. You can identify which points on the plots correspond to which population by clicking on them, or alternatively, by selecting populations on the second table, which contains those used in the PCA computation.

#### 6. f-statistics

Compute f2, f3 and f4 for specific combinations of populations. The statistics f3 and f4 can be assigned to an angle in the interval [0,180] deg. which gives them a scale for comparison purposes between different combinations of populations.

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

from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np



class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent = None, width = 5, height = 4, dpi = 100):
        self.fig = Figure(figsize = (width, height), dpi = dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)



class Plot(QWidget):
    def __init__(self, title, xlabel, ylabel, width, height, dpi):
        QWidget.__init__(self)

        self.canvas = MatplotlibCanvas(self, width, height, dpi)

        self.canvas.axes.set_title(title)

        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)

        self.canvas.fig.tight_layout()

        toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    def plot_fit(self, x, y, alpha, title, xlabel, ylabel):
        self.canvas.axes.clear()

        self.canvas.axes.set_title(title)

        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)

        self.canvas.axes.plot(x, y, '.')
        self.canvas.axes.plot(x, alpha * x)

        self.canvas.fig.canvas.draw()

    def plot_histogram(self, histogram, title, xlabel, ylabel):
        counts = histogram[0]
        edges = histogram[1]

        self.canvas.axes.clear()

        self.canvas.axes.set_title(title)

        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)

        self.canvas.axes.bar(edges[:-1], counts, width = np.diff(edges), edgecolor = 'black', align = 'edge')

        self.canvas.fig.canvas.draw()

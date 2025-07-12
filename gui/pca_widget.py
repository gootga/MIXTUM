from gui.plots import Plot

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QHeaderView, QPushButton, QSizePolicy
from PySide6.QtWidgets import QTableWidgetItem

import numpy as np



class PCAWidget(QWidget):
    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Selected populations table widget
        self.sel_pops_table = QTableWidget()
        self.sel_pops_table.setColumnCount(1)
        self.sel_pops_table.setSortingEnabled(True)
        self.sel_pops_table.verticalHeader().setVisible(False)
        self.sel_pops_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sel_pops_table.setHorizontalHeaderLabels(['Selected populations'])

        # Selected populations PCA table widget
        self.sel_pops_pca_table = QTableWidget()
        self.sel_pops_pca_table.setColumnCount(1)
        self.sel_pops_pca_table.setSortingEnabled(True)
        self.sel_pops_pca_table.verticalHeader().setVisible(False)
        self.sel_pops_pca_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sel_pops_pca_table.setHorizontalHeaderLabels(['PCA populations'])

        self.pca_items = []
        self.pca_indices = {}
        self.pca_names = []

        # Compute button
        self.compute_button = QPushButton('Compute PCA')
        self.compute_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.compute_button.setEnabled(False)

        # PCA Plot
        self.pca_plot = Plot('PCA', 'PC1', 'PC2', 5, 5, 100, projection='3d', zlabel='PC3', multi_selectable=True)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.sel_pops_table)
        controls_layout.addWidget(self.compute_button)

        layout = QHBoxLayout(self)
        layout.addLayout(controls_layout, 1)
        layout.addWidget(self.sel_pops_pca_table, 1)
        layout.addWidget(self.pca_plot, 2)

        # Connections
        self.sel_pops_table.itemSelectionChanged.connect(self.sel_pops_changed)
        self.sel_pops_pca_table.itemSelectionChanged.connect(self.plot_sel_pca_points)
        self.pca_plot.selected_indices_changed.connect(self.select_pops_pca)
        self.compute_button.clicked.connect(self.compute_pca)
        self.compute_button.clicked.connect(self.init_sel_pops_pca_table)

    @Slot(bool)
    def init_sel_pops_table(self, result):
        if not result:
            return

        self.sel_pops_table.clearContents()
        self.sel_pops_table.setRowCount(len(self.core.selected_pops))

        for index, pop in enumerate(self.core.selected_pops):
            item = QTableWidgetItem(pop)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.sel_pops_table.setItem(index, 0, item)

    @Slot()
    def init_sel_pops_pca_table(self):
        self.sel_pops_pca_table.clearContents()
        self.sel_pops_pca_table.setRowCount(len(self.pca_items))

        self.pca_indices = {}
        self.pca_names.clear()

        for index, item in enumerate(self.pca_items):
            pca_item = QTableWidgetItem(item.text())
            pca_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.sel_pops_pca_table.setItem(index, 0, pca_item)

            # Store indices and poop names to link points and pops
            self.pca_indices[pca_item.text()] = index
            self.pca_names.append(pca_item.text())

    @Slot()
    def sel_pops_changed(self):
        self.compute_button.setEnabled(len(self.sel_pops_table.selectedItems()) > 2)

    @Slot(list)
    def select_pops_pca(self, indices):
        self.sel_pops_pca_table.clearSelection()
        for index in indices:
            # pca_item = self.sel_pops_pca_table.item(self.pca_indices[index], 0)
            pca_items = self.sel_pops_pca_table.findItems(self.pca_names[index], Qt.MatchFlag.MatchExactly)
            pca_items[0].setSelected(True)

    @Slot()
    def plot_sel_pca_points(self):
        indices = []
        for item in self.sel_pops_pca_table.selectedItems():
            indices.append(self.pca_indices[item.text()])

        points = np.take(self.core.principal_components, indices, 1)
        self.pca_plot.plot_multiple_selected_points(points, indices)

    @Slot()
    def compute_pca(self):
        self.pca_items = self.sel_pops_table.selectedItems()
        pops = [item.text() for item in self.pca_items]

        self.core.compute_pca(pops)

        xlabel = f"PC1 {self.core.explained_variance[0]:.1f}%"
        ylabel = f"PC2 {self.core.explained_variance[1]:.1f}%"
        zlabel = f"PC3 {self.core.explained_variance[2]:.1f}%"

        self.pca_plot.plot_pca(self.core.principal_components, 'PCA', xlabel, ylabel, zlabel)
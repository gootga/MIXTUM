from gui.plots import Plot

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QHeaderView, QPushButton, QSizePolicy
from PySide6.QtWidgets import QTableWidgetItem



class PCAWidget(QWidget):
    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Auxiliaries table widget
        self.sel_pops_table = QTableWidget()
        self.sel_pops_table.setColumnCount(1)
        self.sel_pops_table.setSortingEnabled(True)
        self.sel_pops_table.verticalHeader().setVisible(False)
        self.sel_pops_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sel_pops_table.setHorizontalHeaderLabels(['Auxiliaries'])

        # Compute button
        self.compute_button = QPushButton('Compute PCA')
        self.compute_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.compute_pca)

        # Connections
        self.sel_pops_table.itemSelectionChanged.connect(self.sel_pops_changed)

        # PCA Plot
        self.pca_plot = Plot('PCA', 'PC1', 'PC2', 5, 5, 100, projection='3d', zlabel='PC3')

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.sel_pops_table)
        controls_layout.addWidget(self.compute_button)

        layout = QHBoxLayout(self)
        layout.addLayout(controls_layout)
        layout.addWidget(self.pca_plot)

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
    def sel_pops_changed(self):
        self.compute_button.setEnabled(len(self.sel_pops_table.selectedItems()) > 2)

    @Slot()
    def compute_pca(self):
        sel_items = self.sel_pops_table.selectedItems()
        pops = [item.text() for item in sel_items]
        pcs = self.core.allele_frequencies_pca(pops)
        self.pca_plot.plot_pca(pcs)
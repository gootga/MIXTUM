from gui.log_system import LogSystem
from gui.open_widget import OpenWidget
from gui.plots import Plot
from gui.worker import Worker

from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtWidgets import QWidget, QTableWidget, QAbstractScrollArea, QAbstractItemView, QTableWidgetItem, QPushButton, QSizePolicy, QProgressBar, QHBoxLayout, QTabWidget, QVBoxLayout, QSplitter, QHeaderView



class MixModelWidget(QWidget):
    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Thread pool
        self.thread_pool = QThreadPool()

        # Log
        self.log = LogSystem(['main'])
        self.log.set_entry('main', 'Choose admixture model and auxiliary populations, then compute results.')

        # Hybrid table widget
        self.hybrid_table = QTableWidget()
        self.hybrid_table.setColumnCount(1)
        self.hybrid_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.hybrid_table.verticalHeader().setVisible(False)
        self.hybrid_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.hybrid_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hybrid_table.setHorizontalHeaderLabels(['Hybrid'])

        # Parent 1 table widget
        self.parent1_table = QTableWidget()
        self.parent1_table.setColumnCount(1)
        self.parent1_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parent1_table.verticalHeader().setVisible(False)
        self.parent1_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.parent1_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parent1_table.setHorizontalHeaderLabels(['Parent 1'])

        # Parent 2 table widget
        self.parent2_table = QTableWidget()
        self.parent2_table.setColumnCount(1)
        self.parent2_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parent2_table.verticalHeader().setVisible(False)
        self.parent2_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.parent2_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parent2_table.setHorizontalHeaderLabels(['Parent 2'])

        # Auxiliaries table widget
        self.aux_table = QTableWidget()
        self.aux_table.setColumnCount(1)
        self.aux_table.verticalHeader().setVisible(False)
        self.aux_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.aux_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.aux_table.setHorizontalHeaderLabels(['Auxiliaries'])

        # Connections
        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

        # Compute button
        self.compute_button = QPushButton('Compute results')
        self.compute_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.compute_results)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(9)
        self.progress_bar.setValue(0)

        # Plots
        self.plot_prime = Plot('Renormalized admixture', 'x', 'y', 5, 4, 100)
        self.plot_std = Plot('Standard admixture', 'x', 'y', 5, 4, 100)
        self.plot_histogram = Plot('Histogram', 'x', 'y', 5, 4, 100)

        # Detach / attach plots panel button
        self.detach_button = QPushButton('Detach plots')
        self.detach_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.detach_button.setCheckable(True)
        self.detach_button.clicked.connect(self.detach_plots)

        # Plots tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.plot_prime, 'Renormalized admixture')
        self.tab_widget.addTab(self.plot_std, 'Standard admixture')
        self.tab_widget.addTab(self.plot_histogram, 'f4 ratio histogram')

        # Plots panel layout
        playout = QVBoxLayout()
        playout.addWidget(self.tab_widget)
        playout.addWidget(self.detach_button, 0, Qt.AlignCenter)

        # Plots panel
        self.plots_panel = OpenWidget()
        self.plots_panel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        self.plots_panel.setLayout(playout)

        # Tables layout
        tlayout = QHBoxLayout()
        tlayout.addWidget(self.hybrid_table)
        tlayout.addWidget(self.parent1_table)
        tlayout.addWidget(self.parent2_table)
        tlayout.addWidget(self.aux_table)

        # Tables widget
        twidget = QWidget()
        twidget.setLayout(tlayout)

        # Splitter
        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.addWidget(twidget)
        self.splitter.addWidget(self.plots_panel)
        #self.splitter.setStretchFactor(0, 1.0)
        #self.splitter.setStretchFactor(1, 7.0)

        # Lower layout
        llayout = QHBoxLayout()
        llayout.addWidget(self.compute_button)
        llayout.addWidget(self.progress_bar)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        layout.addLayout(llayout)

    @Slot(bool)
    def init_pop_tables(self, result):
        if not result:
            return

        self.populate_table_widget(self.hybrid_table)
        self.populate_table_widget(self.parent1_table)
        self.populate_table_widget(self.parent2_table)
        self.populate_table_widget(self.aux_table)

        self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
        self.check_table_selection(self.parent1_table, self.core.parent1_pop)
        self.check_table_selection(self.parent2_table, self.core.parent2_pop)
        self.check_aux_table_selection()

        self.compute_button.setEnabled(True)

    def check_table_selection(self, table, pop):
        self.hybrid_table.itemSelectionChanged.disconnect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.disconnect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.disconnect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.disconnect(self.aux_changed)

        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item.text() == pop:
                item.setSelected(True)
            else:
                item.setSelected(False)

        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

    def check_aux_table_selection(self):
        self.hybrid_table.itemSelectionChanged.disconnect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.disconnect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.disconnect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.disconnect(self.aux_changed)

        for row in range(self.aux_table.rowCount()):
            item = self.aux_table.item(row, 0)
            if item.text() in self.core.aux_pops:
                item.setSelected(True)
            else:
                item.setSelected(False)

        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

    def populate_table_widget(self, table):
        table.clearContents()
        table.setRowCount(len(self.core.selected_pops))

        for index, pop in enumerate(self.core.selected_pops):
            item = QTableWidgetItem(pop)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(index, 0, item)

    @Slot()
    def hybrid_changed(self):
        sel_items = self.hybrid_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_hybrid_pop(sel_items[0].text())

            self.check_table_selection(self.parent1_table, self.core.parent1_pop)
            self.check_table_selection(self.parent2_table, self.core.parent2_pop)
            self.check_aux_table_selection()

    @Slot()
    def parent1_changed(self):
        sel_items = self.parent1_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_parent1_pop(sel_items[0].text())

            self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
            self.check_table_selection(self.parent2_table, self.core.parent2_pop)
            self.check_aux_table_selection()

    @Slot()
    def parent2_changed(self):
        sel_items = self.parent2_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_parent2_pop(sel_items[0].text())

            self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
            self.check_table_selection(self.parent1_table, self.core.parent1_pop)
            self.check_aux_table_selection()

    @Slot()
    def aux_changed(self):
        sel_items = self.aux_table.selectedItems()
        self.core.set_aux_pops([item.text() for item in sel_items])
        self.check_aux_table_selection()
        self.compute_button.setEnabled(len(self.aux_table.selectedItems()) > 0)

    @Slot(str)
    def computation_finished(self, worker_name):
        self.log.set_entry('main', self.core.admixture_data())

        self.plot_prime.plot_fit(self.core.f4ab_prime, self.core.f4xb_prime, self.core.alpha, f'Renormalized admixture: {self.core.hybrid_pop} = alpha {self.core.parent1_pop} + (1 - alpha) {self.core.parent2_pop}', f"f4'({self.core.parent1_pop}, {self.core.parent2_pop}; i, j)", f"f4'({self.core.hybrid_pop}, {self.core.parent2_pop}; i, j)")
        self.plot_std.plot_fit(self.core.f4ab_std, self.core.f4xb_std, self.core.alpha_std, f'Standard admixture: {self.core.hybrid_pop} = alpha {self.core.parent1_pop} + (1 - alpha) {self.core.parent2_pop}', f"f4({self.core.parent1_pop}, {self.core.parent2_pop}; i, j)", f"f4({self.core.hybrid_pop}, {self.core.parent2_pop}; i, j)")
        self.plot_histogram.plot_histogram(self.core.alpha_ratio_hist, f'{self.core.hybrid_pop} = alpha {self.core.parent1_pop} + (1 - alpha) {self.core.parent2_pop}', 'f4 ratio', 'Counts')

    @Slot()
    def compute_results(self):
        worker = Worker('results', self.core.compute_results)
        worker.signals.progress[int].connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.computation_finished)

        self.thread_pool.start(worker)

    @Slot(bool)
    def detach_plots(self, checked):
        if checked:
            self.plots_panel.setParent(None)
            self.plots_panel.setWindowFlag(Qt.WindowCloseButtonHint, False)
            self.plots_panel.show()
        else:
            self.splitter.addWidget(self.plots_panel)
            #self.splitter.setStretchFactor(0, 1.0)
            #self.splitter.setStretchFactor(1, 7.0)

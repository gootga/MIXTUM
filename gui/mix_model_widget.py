from gui.log_system import LogSystem
from gui.open_widget import OpenWidget
from gui.plots import Plot
from gui.worker import Worker

from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtWidgets import QWidget, QTableWidget, QGroupBox, QStackedLayout, QLabel, QSpinBox, QFormLayout
from PySide6.QtWidgets import QAbstractItemView, QTableWidgetItem, QPushButton, QSizePolicy, QProgressBar, QHBoxLayout
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QSplitter, QHeaderView, QFileDialog



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
        self.hybrid_table.setSortingEnabled(True)
        self.hybrid_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.hybrid_table.verticalHeader().setVisible(False)
        self.hybrid_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.hybrid_table.setHorizontalHeaderLabels(['Hybrid'])

        # Parent 1 table widget
        self.parent1_table = QTableWidget()
        self.parent1_table.setColumnCount(1)
        self.parent1_table.setSortingEnabled(True)
        self.parent1_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.parent1_table.verticalHeader().setVisible(False)
        self.parent1_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parent1_table.setHorizontalHeaderLabels(['Parent 1'])

        # Parent 2 table widget
        self.parent2_table = QTableWidget()
        self.parent2_table.setColumnCount(1)
        self.parent2_table.setSortingEnabled(True)
        self.parent2_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.parent2_table.verticalHeader().setVisible(False)
        self.parent2_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parent2_table.setHorizontalHeaderLabels(['Parent 2'])

        # Auxiliaries table widget
        self.aux_table = QTableWidget()
        self.aux_table.setColumnCount(1)
        self.aux_table.setSortingEnabled(True)
        self.aux_table.verticalHeader().setVisible(False)
        self.aux_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.aux_table.setHorizontalHeaderLabels(['Auxiliaries'])

        # Connections
        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

        # Compute button
        self.compute_button = QPushButton('Compute results')
        self.compute_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.compute_results)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(9)
        self.progress_bar.setValue(0)

        # Save f4-points button
        self.save_f4_button = QPushButton('Save f4-points')
        self.save_f4_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.save_f4_button.setEnabled(False)
        self.save_f4_button.clicked.connect(self.save_f4_points)

        # Save results button
        self.save_results_button = QPushButton('Save admixture data')
        self.save_results_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.save_results_button.setEnabled(False)
        self.save_results_button.clicked.connect(self.save_results)

        # Plots
        self.plot_prime = Plot('Renormalized admixture', 'x', 'y', 5, 4, 100, selectable=True)
        self.plot_std = Plot('Standard admixture', 'x', 'y', 5, 4, 100, selectable=True)
        self.plot_histogram = Plot('Histogram', 'x', 'y', 5, 4, 100)
        self.plot_bars = Plot('Hybrid', '', '', 5, 1.25, 100, False, False)
        self.plot_angle = Plot('Angles', '', '', 4, 4, 100, show_toolbar = False, polar = True)

        # Selected pop widgets: prime
        self.prime_sel_pops_label = QLabel()
        prime_sel_pops_form_layout = QFormLayout()
        prime_sel_pops_form_layout.addRow('Selected auxiliaries:', self.prime_sel_pops_label)

        prime_widget = QWidget()
        prime_layout = QVBoxLayout(prime_widget)

        prime_layout.addWidget(self.plot_prime)
        prime_layout.addLayout(prime_sel_pops_form_layout)

        self.plot_prime.selected_index_changed.connect(self.set_prime_sel_pops_label)

        # Selected pop widgets: std
        self.std_sel_pops_label = QLabel()
        std_sel_pops_form_layout = QFormLayout()
        std_sel_pops_form_layout.addRow('Selected auxiliaries:', self.std_sel_pops_label)

        std_widget = QWidget()
        std_layout = QVBoxLayout(std_widget)

        std_layout.addWidget(self.plot_std)
        std_layout.addLayout(std_sel_pops_form_layout)

        self.plot_std.selected_index_changed.connect(self.set_std_sel_pops_label)

        # f4 ratio histogram bins spinbox
        self.bins_spinbox = QSpinBox(minimum = 1, maximum = 1000, value = self.core.alpha_ratio_hist_bins)
        self.bins_spinbox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.bins_spinbox.setEnabled(False)
        self.bins_spinbox.valueChanged.connect(self.compute_histogram)

        # Bins form layout
        bins_form_layout = QFormLayout()
        bins_form_layout.addRow('#Bins:', self.bins_spinbox)

        # f4 ratio histogram widget
        f4_ratio_histogram_widget = QWidget()
        f4_ratio_histogram_layout = QVBoxLayout(f4_ratio_histogram_widget)
        f4_ratio_histogram_layout.addWidget(self.plot_histogram)
        f4_ratio_histogram_layout.addLayout(bins_form_layout)

        # Angles widget
        angles_widget = QWidget()
        angles_layout = QVBoxLayout(angles_widget)
        angles_layout.addWidget(self.plot_angle)

        # Detach / attach plots panel button
        self.detach_button = QPushButton('Detach plots')
        self.detach_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.detach_button.setCheckable(True)
        self.detach_button.clicked.connect(self.detach_plots)

        # Plots tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(prime_widget, 'Renormalized admixture')
        self.tab_widget.addTab(std_widget, 'Standard admixture')
        self.tab_widget.addTab(f4_ratio_histogram_widget, 'f4 ratio histogram')
        self.tab_widget.addTab(angles_widget, 'Angles')

        # Alpha out of range label
        alpha_label = QLabel('Proportions out of range')
        alpha_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        alpha_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        alpha_label.setStyleSheet('color: white; background-color: magenta; font-size: 24pt;')

        # Admixture proportions group box
        plot_bars_groupbox = QGroupBox('Admixture proportions')
        plot_bars_groupbox.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.pbglayout = QStackedLayout(plot_bars_groupbox)
        self.pbglayout.addWidget(self.plot_bars)
        self.pbglayout.addWidget(alpha_label)
        self.pbglayout.setCurrentIndex(0)

        # Plots panel layout
        playout = QVBoxLayout()
        playout.addWidget(self.tab_widget)
        playout.addWidget(plot_bars_groupbox)
        playout.addWidget(self.detach_button, 0, Qt.AlignmentFlag.AlignCenter)

        # Plots panel
        self.plots_panel = OpenWidget()
        self.plots_panel.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
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
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.addWidget(twidget)
        self.splitter.addWidget(self.plots_panel)

        # Lower layout
        llayout = QHBoxLayout()
        llayout.addWidget(self.compute_button)
        llayout.addWidget(self.progress_bar)
        llayout.addWidget(self.save_f4_button)
        llayout.addWidget(self.save_results_button)

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
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
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
        self.plot_angle.plot_angle('Angles', [self.core.angle_pre_jl, self.core.angle_post_jl])

        if 0 <= self.core.alpha <= 1:
            self.plot_bars.plot_bars(self.core.hybrid_pop, self.core.parent1_pop, self.core.parent2_pop, self.core.alpha)
            self.pbglayout.setCurrentIndex(0)
        else:
            self.pbglayout.setCurrentIndex(1)

        self.save_f4_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
        self.bins_spinbox.setEnabled(True)

    @Slot()
    def compute_results(self):
        worker = Worker('results', self.core.compute_results)
        worker.signals.progress[int].connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.computation_finished)

        self.thread_pool.start(worker)

    @Slot(int)
    def compute_histogram(self, bins):
        self.core.compute_f4_ratio_histogram(bins)
        self.plot_histogram.plot_histogram(self.core.alpha_ratio_hist, f'{self.core.hybrid_pop} = alpha {self.core.parent1_pop} + (1 - alpha) {self.core.parent2_pop}', 'f4 ratio', 'Counts')

    @Slot(bool)
    def detach_plots(self, checked):
        if checked:
            self.detach_button.setText('Attach plots')
            self.plots_panel.setParent(None)
            self.plots_panel.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
            self.plots_panel.show()
        else:
            self.detach_button.setText('Detach plots')
            self.splitter.addWidget(self.plots_panel)

    @Slot()
    def save_f4_points(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)

        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            self.core.save_f4_points(file_path)

    @Slot()
    def save_results(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)

        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            self.core.save_admixture_data(file_path)

    @Slot(int)
    def set_prime_sel_pops_label(self, index):
        pop1, pop2 = self.core.get_aux_pop_pair(index)
        self.prime_sel_pops_label.setText(f"{pop1} + {pop2}")

    @Slot(int)
    def set_std_sel_pops_label(self, index):
        pop1, pop2 = self.core.get_aux_pop_pair(index)
        self.std_sel_pops_label.setText(f"{pop1} + {pop2}")
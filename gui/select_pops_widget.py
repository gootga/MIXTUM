from gui.log_system import LogSystem
from gui.searchable_table_widget import SearchableTableWidget
from gui.worker import Worker

from PySide6.QtCore import Signal, Slot, QThreadPool
from PySide6.QtWidgets import QWidget, QTableWidget, QAbstractScrollArea, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QProgressBar, QVBoxLayout, QHBoxLayout


class SelectPopsWidget(QWidget):
    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Thread pool
        self.thread_pool = QThreadPool()

        # Log
        self.log = LogSystem(['main', 'progress'])

        # Searchable table containing available populations
        self.search_widget = SearchableTableWidget()

        # Table containing selected populations
        self.selected_table = QTableWidget()
        self.selected_table.setColumnCount(1)
        self.selected_table.verticalHeader().setVisible(False)
        self.selected_table.horizontalHeader().setVisible(False)
        self.selected_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        # Select populations button
        self.select_button = QPushButton('Select populations')
        self.select_button.clicked.connect(self.select_populations)

        # Remove populations button
        self.remove_button = QPushButton('Deselect populations')
        self.remove_button.clicked.connect(self.remove_populations)

        # Reset populations button
        self.reset_button = QPushButton('Reset populations')
        self.reset_button.clicked.connect(self.reset_populations)

        # Number of computing processes
        num_procs_label = QLabel('Number of processes')
        self.num_procs_spin_box = QSpinBox()
        self.num_procs_spin_box.setMinimum(1)
        self.num_procs_spin_box.setMaximum(9999)
        self.num_procs_spin_box.valueChanged.connect(self.core.set_num_procs)
        self.num_procs_spin_box.setValue(1)

        # Compute allele frequencies files button
        self.comp_button = QPushButton('Compute frequencies')
        self.comp_button.clicked.connect(self.compute_frequencies)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)

        # Select table buttons layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.remove_button)
        hlayout.addWidget(self.reset_button)

        # Select table layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.selected_table)
        vlayout.addLayout(hlayout)

        # Search table layout
        slayout = QVBoxLayout()
        slayout.addWidget(self.search_widget)
        slayout.addWidget(self.select_button)

        # Tables layout
        tlayout = QHBoxLayout()
        tlayout.addLayout(slayout)
        tlayout.addLayout(vlayout)

        # Layout
        layout = QVBoxLayout(self)
        layout.addLayout(tlayout)
        layout.addWidget(num_procs_label)
        layout.addWidget(self.num_procs_spin_box)
        layout.addWidget(self.comp_button)
        layout.addWidget(self.progress_bar)

    @Slot()
    def init_search_table(self):
        self.search_widget.init_table(self.core.avail_pops)

    @Slot()
    def init_selected_table(self):
        self.populate_selected_table(self.core.parsed_pops)

    @Slot()
    def select_populations(self):
        self.core.append_pops(self.search_widget.selected_items())
        self.populate_selected_table(self.core.selected_pops)

    @Slot()
    def remove_populations(self):
        self.core.remove_pops([item.text() for item in self.selected_table.selectedItems()])
        self.populate_selected_table(self.core.selected_pops)

    @Slot()
    def reset_populations(self):
        self.core.reset_pops()
        self.populate_selected_table(self.core.selected_pops)

    @Slot(object)
    def populate_selected_table(self, pops):
        self.selected_table.clearContents()
        self.selected_table.setRowCount(len(pops))

        for index, item in enumerate(pops):
            table_widget_item = QTableWidgetItem(item)
            self.selected_table.setItem(index, 0, table_widget_item)

    def set_log_text(self):
        text_list = [text for key, text in self.log_text_lines.items() if len(text) > 0]
        self.log_text = '\n'.join(text_list)
        self.log_text_changed.emit(self.log_text)

    @Slot(float)
    def set_progress_log_text(self, percent_done):
        self.log.set_entry('progress', f'{percent_done:.1%}')

    @Slot(str)
    def computing_finished(self, worker_name):
        self.log.set_entry('main', 'Computation finished.')

    @Slot()
    def compute_frequencies(self):
        self.log.clear_entry('progress')
        self.log.set_entry('main', f'Computing allele frequencies...')

        self.progress_bar.setMaximum(len(self.core.selected_pops))

        worker = Worker('freqs', self.core.parallel_compute_populations_frequencies)
        worker.signals.progress[float].connect(self.set_progress_log_text)
        worker.signals.progress[int].connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.computing_finished)

        self.thread_pool.start(worker)

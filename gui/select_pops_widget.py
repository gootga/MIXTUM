from gui.log_system import LogSystem
from gui.worker import Worker

from PySide6.QtCore import Signal, Slot, QThreadPool
from PySide6.QtWidgets import QWidget, QPushButton, QSpinBox, QVBoxLayout


class SelectPopsWidget(QWidget):
    log_text_changed = Signal(str)

    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Thread pool
        self.thread_pool = QThreadPool()

        # Log text
        self.log = LogSystem(['main', 'progress'])

        # Number of computing processes
        self.num_procs_spin_box = QSpinBox()
        self.num_procs_spin_box.setMinimum(1)
        self.num_procs_spin_box.setMaximum(9999)
        self.num_procs_spin_box.valueChanged.connect(self.core.set_num_procs)
        self.num_procs_spin_box.setValue(1)

        # Compute allele frequencies files button
        self.comp_button = QPushButton('Compute frequencies')
        self.comp_button.clicked.connect(self.compute_frequencies)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.num_procs_spin_box)
        layout.addWidget(self.comp_button)

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

        worker = Worker('freqs', self.core.parallel_compute_populations_frequencies)
        worker.signals.progress[float].connect(self.set_progress_log_text)
        worker.signals.finished.connect(self.computing_finished)

        self.thread_pool.start(worker)

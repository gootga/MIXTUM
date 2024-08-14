from pathlib import Path

from gui.log_system import LogSystem
from gui.select_file_widget import SelectFileWidget
from gui.worker import Worker

from PySide6.QtCore import Signal, Slot, QThreadPool
from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout



class InputFilesWidget(QWidget):
    ind_file_parsed = Signal()
    pops_file_parsed = Signal()

    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Thread pool
        self.thread_pool = QThreadPool()
        self.worker_finished = {'geno': False, 'ind': False, 'snp': False}

        # Log system
        self.log = LogSystem(['main', 'geno', 'ind', 'snp', 'pops', 'check'])

        # Select file widgets
        self.geno_file_widget = SelectFileWidget('Select .geno file', '(*.geno)')
        self.ind_file_widget = SelectFileWidget('Select .ind file', '(*.ind)')
        self.snp_file_widget = SelectFileWidget('Select .snp file', '(*.snp)')
        self.pops_file_widget = SelectFileWidget('Select populations file', None)

        self.geno_file_widget.file_path_selected.connect(self.core.set_geno_file_path)
        self.ind_file_widget.file_path_selected.connect(self.core.set_ind_file_path)
        self.snp_file_widget.file_path_selected.connect(self.core.set_snp_file_path)
        self.pops_file_widget.file_path_selected.connect(self.core.set_pops_file_path)

        # Check files button
        self.check_button = QPushButton('Parse and check')
        self.check_button.setEnabled(False)
        self.core.input_file_paths_state.connect(self.check_button.setEnabled)
        self.check_button.clicked.connect(self.check_input_files)

        # Check error
        self.core.geno_file_error.connect(self.geno_check_failed)
        self.core.ind_file_error.connect(self.ind_check_failed)
        self.core.snp_file_error.connect(self.snp_check_failed)

        # Parse selected pops file button
        self.parse_pops_button = QPushButton('Load selected populations')
        self.parse_pops_button.setEnabled(False)
        self.core.pops_file_path_state.connect(self.parse_pops_button.setEnabled)
        self.parse_pops_button.clicked.connect(self.parse_pops_file)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.geno_file_widget)
        layout.addWidget(self.ind_file_widget)
        layout.addWidget(self.snp_file_widget)
        layout.addWidget(self.pops_file_widget)
        layout.addWidget(self.check_button)
        layout.addWidget(self.parse_pops_button)

    @Slot(int)
    def set_geno_log_text(self, num_rows):
        self.log.set_entry('geno', f'Number of rows in .geno file: {num_rows}')

    @Slot(int)
    def set_ind_log_text(self, num_rows):
        self.log.set_entry('ind', f'Number of rows in .ind file: {num_rows}')

    @Slot(int)
    def set_snp_log_text(self, num_rows):
        self.log.set_entry('snp', f'Number of rows in .snp file: {num_rows}')

    @Slot(int)
    def set_pops_log_text(self, num_rows):
        self.log.set_entry('pops', f'Number of selected populations: {num_rows}')

    @Slot(str)
    def checking_finished(self, worker_name):
        self.worker_finished[worker_name] = True

        if all([status for name, status in self.worker_finished.items()]):
            if self.core.check_input_files():
                self.log.set_entry('main', 'Checking finished.')
                self.log.append_entry('check', 'Parsed input files seem to have a valid structure.')

        if worker_name == 'ind':
            self.ind_file_parsed.emit()

    @Slot()
    def geno_check_failed(self):
        self.log.set_entry('main', 'Checking error!')
        self.log.append_entry('check', 'Error in .geno file: not all rows have the same number of columns.')

    @Slot(int, int)
    def ind_check_failed(self, num_pops, num_cols):
        self.log.set_entry('main', 'Checking error!')
        self.log.append_entry('check', f'Error: Number of populations ({num_pops}) in .ind file is not equal to number of columns ({num_cols}) in .geno file.')

    @Slot(int, int)
    def snp_check_failed(self, num_alleles, num_rows):
        self.log.set_entry('main', 'Checking error!')
        self.log.append_entry('check', f'Error: Number of alleles ({num_alleles}) in .snp file is not equal to number of rows ({num_rows}) in .geno file.')

    @Slot()
    def check_input_files(self):
        self.log.clear_entry('check')
        self.log.set_entry('main', 'Checking input files...')

        self.worker_finished = {'geno': False, 'ind': False, 'snp': False}

        geno_worker = Worker('geno', self.core.geno_table_shape)
        geno_worker.signals.progress[int].connect(self.set_geno_log_text)
        geno_worker.signals.finished.connect(self.checking_finished)

        ind_worker = Worker('ind', self.core.parse_ind_file)
        ind_worker.signals.progress[int].connect(self.set_ind_log_text)
        ind_worker.signals.finished.connect(self.checking_finished)

        snp_worker = Worker('snp', self.core.parse_snp_file)
        snp_worker.signals.progress[int].connect(self.set_snp_log_text)
        snp_worker.signals.finished.connect(self.checking_finished)

        self.thread_pool.start(geno_worker)
        self.thread_pool.start(ind_worker)
        self.thread_pool.start(snp_worker)

    @Slot(str)
    def parsing_finished(self, worker_name):
        self.log.set_entry('main', 'Parsing finished.')
        self.pops_file_parsed.emit()

    @Slot()
    def parse_pops_file(self):
        self.log.set_entry('main', 'Parsing selected populations file...')

        pops_worker = Worker('pops', self.core.parse_selected_populations)
        pops_worker.signals.progress[int].connect(self.set_pops_log_text)
        pops_worker.signals.finished.connect(self.parsing_finished)

        self.thread_pool.start(pops_worker)

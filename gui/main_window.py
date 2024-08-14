from gui.core import Core
from gui.log_widget import LogWidget
from gui.input_files_widget import InputFilesWidget
from gui.select_pops_widget import SelectPopsWidget

from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout



class MainWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('Mixtum')

        # Core
        self.core = Core()

        # Log widget
        self.log_widget = LogWidget()

        # Input files widget
        self.input_files_widget = InputFilesWidget(self.core)
        self.input_files_widget.log_text_changed.connect(self.log_widget.set_text)

        # Select pops widget
        self.sel_pops_widget = SelectPopsWidget(self.core)
        self.sel_pops_widget.log_text_changed.connect(self.log_widget.set_text)

        # Tab widget
        self.tab = QTabWidget()
        self.tab.addTab(self.input_files_widget, 'Input files')
        self.tab.addTab(self.sel_pops_widget, 'Select populations')

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.tab)
        layout.addWidget(self.log_widget)

        # Window dimensions

        #geometry = self.screen().availableGeometry()
        #self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)

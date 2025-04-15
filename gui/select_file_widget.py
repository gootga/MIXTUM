from PySide6.QtCore import Signal, Slot, QSize
from PySide6.QtWidgets import QWidget, QPushButton, QSizePolicy, QVBoxLayout, QFileDialog



class SelectFileWidget(QWidget):
    file_path_selected = Signal(str)

    def __init__(self, button_label, name_filter, stylesheet):
        QWidget.__init__(self)

        # File name filter
        self.name_filter = name_filter

        # Select path button
        select_button = QPushButton(button_label)
        select_button.setMinimumSize(QSize(500, 100))
        select_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        select_button.setStyleSheet(stylesheet)
        select_button.clicked.connect(self.select_file_path)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(select_button)

    @Slot()
    def select_file_path(self):
        # File dialog to select file path
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        if self.name_filter is not None:
            dialog.setNameFilter(self.name_filter)

        # Obtain file name
        file_names = []
        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            self.file_path_selected.emit(file_path)

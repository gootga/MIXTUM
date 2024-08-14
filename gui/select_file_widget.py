from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QFrame, QHBoxLayout, QFileDialog



class SelectFileWidget(QWidget):
    file_path_selected = Signal(str)

    def __init__(self, button_label, name_filter):
        QWidget.__init__(self)

        # File name filter
        self.name_filter = name_filter

        # Select path button
        select_button = QPushButton(button_label)
        select_button.clicked.connect(self.select_file_path)

        # Path label
        self.path_label = QLabel()
        self.path_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.path_label.setText('Please, select input file')

        # Layout
        layout = QHBoxLayout(self)
        layout.addWidget(select_button)
        layout.addWidget(self.path_label)

    @Slot()
    def select_file_path(self):
        # File dialog to select file path
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        if self.name_filter is not None:
            dialog.setNameFilter(self.name_filter)

        file_names = []
        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            self.path_label.setText(file_path)
            self.file_path_selected.emit(file_path)

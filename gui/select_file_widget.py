#    Mixtum: the geometry of admixture in population genetics.
#    Copyright (C) 2025  Jose Maria Castelo Ares
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
        select_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        select_button.setStyleSheet(stylesheet)
        select_button.clicked.connect(self.select_file_path)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(select_button)

    @Slot()
    def select_file_path(self):
        # File dialog to select file path
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if self.name_filter is not None:
            dialog.setNameFilter(self.name_filter)

        # Obtain file name
        file_names = []
        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            self.file_path_selected.emit(file_path)

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

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget, QTextEdit, QVBoxLayout



class LogWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Log text edit
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFontFamily('Monospace')

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.log_text_edit)

    @Slot(str)
    def set_text(self, text):
        self.log_text_edit.setPlainText(text)

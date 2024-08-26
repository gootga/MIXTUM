from PySide6.QtCore import Slot
from PySide6.QtWidgets import QWidget, QTextEdit, QVBoxLayout



class LogWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Log text edit
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFontFamily('Courier')

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.log_text_edit)

    @Slot(str)
    def set_text(self, text):
        self.log_text_edit.setPlainText(text)

from PySide6.QtWidgets import QWidget


class OpenWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        pass

    def closeEvent(self, event):
            event.ignore()

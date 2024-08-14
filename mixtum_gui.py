import sys

from gui.main_window import MainWindow

from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


import sys
from multiprocessing import freeze_support

from gui.main_window import MainWindow

from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    freeze_support()
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

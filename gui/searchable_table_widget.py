from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView



class SearchableTableWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Table containing all items
        self.table = []

        # Table containing search results
        #self.results_table = []

        # Search input edit
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('Search population...')
        self.search_edit.textChanged.connect(self.search_table)

        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setSortingEnabled(True)
        self.table_widget.setColumnCount(1)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.horizontalHeader().setVisible(True)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_widget.setHorizontalHeaderLabels(['Available populations'])

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.table_widget)

    @Slot(object)
    def init_table(self, initial_table):
        self.table = [item for item in initial_table]
        self.populate_table_widget(self.table)

    def populate_table_widget(self, table):
        self.table_widget.clearContents()
        self.table_widget.setRowCount(len(table))

        for index, item in enumerate(table):
            table_widget_item = QTableWidgetItem(item)
            table_widget_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table_widget.setItem(index, 0, table_widget_item)

        # self.table_widget.resizeColumnsToContents()

    @Slot(str)
    def search_table(self, text):
        if len(text) > 1:
            results_table = [item for item in self.table if text.lower() in item.lower()]
            self.populate_table_widget(results_table)
        else:
            self.populate_table_widget(self.table)

    def selected_items(self):
        return [item.text() for item in self.table_widget.selectedItems()]

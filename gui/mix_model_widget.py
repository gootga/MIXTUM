from gui.log_system import LogSystem

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QTableWidget, QAbstractScrollArea, QAbstractItemView, QTableWidgetItem, QGridLayout



class MixModelWidget(QWidget):
    def __init__(self, core):
        QWidget.__init__(self)

        # Core
        self.core = core

        # Log
        self.log = LogSystem(['main'])
        self.log.set_entry('main', 'Choose admixture model and auxiliary populations, then compute results.')

        # Hybrid table widget
        self.hybrid_table = QTableWidget()
        self.hybrid_table.setColumnCount(1)
        self.hybrid_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.hybrid_table.verticalHeader().setVisible(False)
        self.hybrid_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.hybrid_table.setHorizontalHeaderLabels(['Hybrid'])

        # Parent 1 table widget
        self.parent1_table = QTableWidget()
        self.parent1_table.setColumnCount(1)
        self.parent1_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parent1_table.verticalHeader().setVisible(False)
        self.parent1_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.parent1_table.setHorizontalHeaderLabels(['Parent 1'])


        # Parent 2 table widget
        self.parent2_table = QTableWidget()
        self.parent2_table.setColumnCount(1)
        self.parent2_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parent2_table.verticalHeader().setVisible(False)
        self.parent2_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.parent2_table.setHorizontalHeaderLabels(['Parent 2'])

        # Auxiliaries table widget
        self.aux_table = QTableWidget()
        self.aux_table.setColumnCount(1)
        self.aux_table.verticalHeader().setVisible(False)
        self.aux_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.aux_table.setHorizontalHeaderLabels(['Auxiliaries'])

        # Connections
        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

        # Layout
        layout = QGridLayout(self)
        layout.addWidget(self.hybrid_table, 0, 0)
        layout.addWidget(self.parent1_table, 0, 1)
        layout.addWidget(self.parent2_table, 0, 2)
        layout.addWidget(self.aux_table, 0, 3)

    @Slot(bool)
    def init_pop_tables(self, result):
        if not result:
            return

        self.populate_table_widget(self.hybrid_table)
        self.populate_table_widget(self.parent1_table)
        self.populate_table_widget(self.parent2_table)
        self.populate_table_widget(self.aux_table)

        self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
        self.check_table_selection(self.parent1_table, self.core.parent1_pop)
        self.check_table_selection(self.parent2_table, self.core.parent2_pop)
        self.check_aux_table_selection()

    def check_table_selection(self, table, pop):
        self.hybrid_table.itemSelectionChanged.disconnect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.disconnect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.disconnect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.disconnect(self.aux_changed)

        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item.text() == pop:
                item.setSelected(True)
            else:
                item.setSelected(False)

        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

    def check_aux_table_selection(self):
        self.hybrid_table.itemSelectionChanged.disconnect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.disconnect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.disconnect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.disconnect(self.aux_changed)

        for row in range(self.aux_table.rowCount()):
            item = self.aux_table.item(row, 0)
            if item.text() in self.core.aux_pops:
                item.setSelected(True)
            else:
                item.setSelected(False)

        self.hybrid_table.itemSelectionChanged.connect(self.hybrid_changed)
        self.parent1_table.itemSelectionChanged.connect(self.parent1_changed)
        self.parent2_table.itemSelectionChanged.connect(self.parent2_changed)
        self.aux_table.itemSelectionChanged.connect(self.aux_changed)

    def populate_table_widget(self, table):
        table.clearContents()
        table.setRowCount(len(self.core.selected_pops))

        for index, pop in enumerate(self.core.selected_pops):
            item = QTableWidgetItem(pop)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(index, 0, item)

    @Slot()
    def hybrid_changed(self):
        sel_items = self.hybrid_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_hybrid_pop(sel_items[0].text())

            self.check_table_selection(self.parent1_table, self.core.parent1_pop)
            self.check_table_selection(self.parent2_table, self.core.parent2_pop)
            self.check_aux_table_selection()

    @Slot()
    def parent1_changed(self):
        sel_items = self.parent1_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_parent1_pop(sel_items[0].text())

            self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
            self.check_table_selection(self.parent2_table, self.core.parent2_pop)
            self.check_aux_table_selection()

    @Slot()
    def parent2_changed(self):
        sel_items = self.parent2_table.selectedItems()
        if len(sel_items) > 0:
            self.core.set_parent2_pop(sel_items[0].text())

            self.check_table_selection(self.hybrid_table, self.core.hybrid_pop)
            self.check_table_selection(self.parent1_table, self.core.parent1_pop)
            self.check_aux_table_selection()

    @Slot()
    def aux_changed(self):
        sel_items = self.aux_table.selectedItems()
        self.core.set_aux_pops([item.text() for item in sel_items])
        self.check_aux_table_selection()

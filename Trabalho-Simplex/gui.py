from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, \
    QComboBox, QPushButton, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import Qt
import sympy
import numpy as np
import simplex as sp
import sys

M = sympy.Symbol('M', positive=True)
HEADER_SPACE = 11
global simplex


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data, header):
        super(TableModel, self).__init__()
        self._data = data
        self.header = header

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return " " * HEADER_SPACE + " " + str(self.header[section]) + " " * HEADER_SPACE
        return super().headerData(section, orientation, role)


class Initial(QMainWindow):
    def __init__(self):
        super(Initial, self).__init__()
        self.CONSTRAINT_EQUALITY_SIGNS = [u"\u2264", u"\u2265", "="]
        self.new_widgets = []
        self.num_lines = 2
        self.num_columns = 2

        self.create_ui()
        self.set_ui_layout()

        self.setFixedWidth(self.sizeHint().width() + 100)
        self.setFixedHeight(self.sizeHint().height() + 100)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

    def create_ui(self):
        self.objective_function_label = QLabel("Função Objetiva", self)
        self.objective_function_label.setFixedHeight(self.objective_function_label.sizeHint().height())
        self.objective_fxn_table = self.create_table(1, 4, ["="], self.create_header_labels(2))

        z_item = QTableWidgetItem("Z")
        self.objective_fxn_table.setItem(0, 3, z_item)
        z_item.setFlags(Qt.ItemIsEnabled)

        self.objective_fxn_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.objective_fxn_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.objective_fxn_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.objective_fxn_table.resizeColumnsToContents()
        self.objective_fxn_table.setFixedHeight(
            self.objective_fxn_table.verticalHeader().length() + self.objective_fxn_table.horizontalHeader().height() +
            self.objective_fxn_table.horizontalScrollBar().height())

        self.constraints_label = QLabel("Restrições", self)
        self.constraints_label.setFixedHeight(self.constraints_label.sizeHint().height())
        self.constraint_table = self.create_table(2, 4, self.CONSTRAINT_EQUALITY_SIGNS, self.create_header_labels(2))
        self.constraint_table.setFixedHeight(self.constraint_table.sizeHint().height())

        self.answers_label = QLabel()

        self.add_row_btn = QPushButton('Add Linha', self)
        self.add_row_btn.clicked.connect(self.add_row_event)
        self.add_col_btn = QPushButton('Add Coluna', self)
        self.add_col_btn.clicked.connect(self.add_column_event)
        self.del_row_btn = QPushButton("Deletar Linha", self)
        self.del_row_btn.clicked.connect(self.del_row_event)
        self.del_col_btn = QPushButton("Deletar Coluna", self)
        self.del_col_btn.clicked.connect(self.del_col_event)
        self.solve_btn = QPushButton('Solver', self)

        self.solve_btn.clicked.connect(self.solve_event)

        self.operation_combo = QComboBox()
        for item in ["Maximize", "Minimize"]:
            self.operation_combo.addItem(item)

    def set_ui_layout(self):
        vbox_layout1 = QHBoxLayout(self)
        self.vbox_layout2 = QVBoxLayout(self)

        vbox_layout1.addWidget(self.add_row_btn)
        vbox_layout1.addWidget(self.add_col_btn)
        vbox_layout1.addWidget(self.del_row_btn)
        vbox_layout1.addWidget(self.del_col_btn)
        vbox_layout1.addWidget(self.operation_combo)
        vbox_layout1.addWidget(self.solve_btn)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_v_layout = QVBoxLayout(self)
        central_widget.setLayout(main_v_layout)

        self.vbox_layout2.addWidget(self.objective_function_label)
        self.vbox_layout2.addWidget(self.objective_fxn_table)
        self.vbox_layout2.addWidget(self.constraints_label)
        self.vbox_layout2.addWidget(self.constraint_table)
        self.vbox_layout2.addWidget(self.answers_label)

        main_v_layout.addLayout(vbox_layout1)
        main_v_layout.addLayout(self.vbox_layout2)

    def create_table(self, rows, cols, equality_signs=None, horizontal_headers=None, vertical_headers=None):
        table = QTableWidget(self)
        table.setColumnCount(cols)
        table.setRowCount(rows)

        if horizontal_headers:
            table.setHorizontalHeaderLabels(horizontal_headers)

        if vertical_headers:
            table.setVerticalHeaderLabels(vertical_headers)

        if equality_signs:
            numofrows = table.rowCount()
            numofcols = table.columnCount()

            for index in range(numofrows):
                equality_signs_combo = QComboBox()
                for item in equality_signs:
                    equality_signs_combo.addItem(item)
                table.setCellWidget(index, numofcols - 2, equality_signs_combo)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        return table

    def create_header_labels(self, num_of_variables):
        """nomeie as colunas para as tabelas x1,x2,.... de um espaco e add resultado"""
        header_labels = [" " * HEADER_SPACE + "x" + str(i + 1) + " " * HEADER_SPACE for i in range(num_of_variables)]
        header_labels.extend([" " * HEADER_SPACE, " " * HEADER_SPACE + "resultado" + " " * HEADER_SPACE])
        return header_labels

    def del_row_event(self):
        if self.constraint_table.rowCount() > 1:
            self.num_lines -= 1
            self.constraint_table.removeRow(self.constraint_table.rowCount() - 1)

    def del_col_event(self):
        if self.constraint_table.columnCount() > 4:
            self.num_columns -= 1
            self.constraint_table.removeColumn(self.constraint_table.columnCount() - 3)
            self.objective_fxn_table.removeColumn(self.objective_fxn_table.columnCount() - 3)

    def add_column_event(self):
        self.num_columns += 1
        self.constraint_table.insertColumn(self.constraint_table.columnCount() - 2)
        self.objective_fxn_table.insertColumn(self.objective_fxn_table.columnCount() - 2)
        self.constraint_table.setHorizontalHeaderLabels(
            self.create_header_labels(self.constraint_table.columnCount() - 2))
        self.objective_fxn_table.setHorizontalHeaderLabels(
            self.create_header_labels(self.constraint_table.columnCount() - 2))

        self.objective_fxn_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.objective_fxn_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.objective_fxn_table.setFixedHeight(
            self.objective_fxn_table.verticalHeader().length() + self.objective_fxn_table.horizontalHeader().height() +
            self.objective_fxn_table.horizontalScrollBar().height() + 10)

    def add_row_event(self):
        self.num_lines += 1
        self.constraint_table.insertRow(self.constraint_table.rowCount())
        equality_signs_combo = QComboBox()
        for item in self.CONSTRAINT_EQUALITY_SIGNS:
            equality_signs_combo.addItem(item)
        self.constraint_table.setCellWidget(self.constraint_table.rowCount() - 1,
                                            self.constraint_table.columnCount() - 2, equality_signs_combo)
        self.constraint_table.resizeRowsToContents()

    def solve_event(self):
        global simplex
        matrix = np.zeros([self.num_lines + 1, self.num_columns + 2], dtype=float)
        matrix[0][0] = self.operation_combo.currentIndex()
        for column in range(0, self.num_columns):
            item = self.objective_fxn_table.item(0, column)
            if item:
                matrix[0][column + 1] = item.text()
            else:
                matrix[0][column + 1] = 0

        for line in range(0, self.num_lines):
            for column in range(0, self.num_columns):
                item = self.constraint_table.item(line, column)
                if item:
                    matrix[line + 1][column + 1] = item.text()
                else:
                    matrix[line + 1][column + 1] = 0
            combo = self.constraint_table.indexWidget(self.constraint_table.model().index(line, self.num_columns))
            matrix[line + 1][0] = combo.currentIndex() + 1
            item = self.constraint_table.item(line, self.num_columns + 1)
            if item:
                matrix[line + 1][self.num_columns + 1] = item.text()
            else:
                matrix[line + 1][self.num_columns + 1] = 0
        simplex = sp.Simplex(matrix)
        simplex.execute()
        solution = Solution(0)
        widget.addWidget(solution)
        widget.setCurrentWidget(solution)


class Solution(QMainWindow):
    def __init__(self, algorithm_index):
        super(Solution, self).__init__()
        self.algorithm_index = algorithm_index
        self.new_widgets = []
        self.num_lines = 2
        self.num_columns = 2

        self.create_ui()
        self.set_ui_layout()

        self.setFixedWidth(self.sizeHint().width() + 400)
        self.setFixedHeight(self.sizeHint().height())
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

    def create_ui(self):
        self.algorithm_label = QLabel("Algoritmo", self)
        self.algorithm_label.setFixedHeight(self.algorithm_label.sizeHint().height())
        self.algorithm_table = self.create_table()

        self.algorithm_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.algorithm_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.algorithm_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.algorithm_table.resizeColumnsToContents()
        self.algorithm_table.setFixedHeight(
            self.algorithm_table.verticalHeader().length() + self.algorithm_table.horizontalHeader().height() +
            self.algorithm_table.horizontalScrollBar().height() - 15)

        self.vb_label = QLabel("VBs", self)
        self.vb_label.setFixedHeight(self.vb_label.sizeHint().height())
        self.vb_table = self.create_vb_line()
        self.vb_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.vb_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.vb_table.setFixedHeight(
            self.vb_table.verticalHeader().length() + self.vb_table.horizontalHeader().height() +
            self.algorithm_table.horizontalScrollBar().height() - 15)

        self.new_simplex_btn = QPushButton('Novo Simplex', self)
        self.new_simplex_btn.clicked.connect(self.new_simplex_event)
        self.previous_algorithm_btn = QPushButton("Anterior", self)
        self.previous_algorithm_btn.clicked.connect(self.previous_algorithm_event)
        self.next_algorithm_btn = QPushButton("Próximo", self)
        self.next_algorithm_btn.clicked.connect(self.next_algorithm_event)

        self.z_label = QLabel("Z")
        self.z_box = self.create_z()
        self.z_box.setReadOnly(True)

    def set_ui_layout(self):
        vbox_layout1 = QHBoxLayout(self)
        vbox_layout2 = QVBoxLayout(self)
        vbox_layout1.addWidget(self.new_simplex_btn)
        vbox_layout1.addWidget(self.previous_algorithm_btn)
        vbox_layout1.addWidget(self.next_algorithm_btn)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_v_layout = QVBoxLayout(self)
        central_widget.setLayout(main_v_layout)

        vbox_layout2.addWidget(self.algorithm_label)
        vbox_layout2.addWidget(self.algorithm_table)
        vbox_layout2.addWidget(self.vb_label)
        vbox_layout2.addWidget(self.vb_table)

        vbox_layout2.addWidget(self.z_label)
        vbox_layout2.addWidget(self.z_box)

        main_v_layout.addLayout(vbox_layout1)
        main_v_layout.addLayout(vbox_layout2)

    def create_table(self):
        global simplex
        header = simplex.get_header()
        table = QtWidgets.QTableView(self)
        model = TableModel(simplex.get_algorithm(self.algorithm_index), header)
        table.setModel(model)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        return table

    def create_vb_line(self):
        global simplex
        vb = simplex.get_vb(self.algorithm_index)
        columns = len(vb)
        vb_line = QTableWidget(self)
        vb_line.setColumnCount(columns)
        vb_line.setRowCount(1)
        vb_line.setHorizontalHeaderLabels([" " * HEADER_SPACE + i + " " * HEADER_SPACE for i in vb])
        for i, value in enumerate(vb.values()):
            vb_line.setItem(0, i, QTableWidgetItem(str(value)))
        vb_line.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        return vb_line

    def new_simplex_event(self):
        widget.setCurrentWidget(initial)

    def previous_algorithm_event(self):
        if self.algorithm_index > 0:
            self.algorithm_index -= 1
        self.reload()

    def next_algorithm_event(self):
        global simplex
        if self.algorithm_index < len(simplex.algorithms) - 1:
            self.algorithm_index += 1
        self.reload()

    def reload(self):
        solution = widget.widget(1)
        widget.removeWidget(solution)
        solution = Solution(self.algorithm_index)
        widget.addWidget(solution)
        widget.setCurrentWidget(solution)

    def create_z(self):
        z = QLineEdit(str(simplex.get_z(self.algorithm_index)))
        return z


app = QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()

initial = Initial()
widget.addWidget(initial)

widget.setWindowTitle("Calculadora-Simplex")
widget.setWindowIcon(QtGui.QIcon("icon.ico"))

widget.setCurrentWidget(initial)

widget.show()
app.exec_()

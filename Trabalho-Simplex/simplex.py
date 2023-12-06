import numpy as np
from collections import OrderedDict


class Simplex:

    def __init__(self, matrix_problem, fo_min=False):
        mp_num_line, mp_num_column = matrix_problem.shape
        self.num_var = mp_num_column - 2
        self.num_rest = mp_num_line - 1
        self.fo_min = False
        self.xf = []
        self.n_xf = []
        self.a = []
        self.a_value = 0
        self.read_config_column(matrix_problem[:, 0])
        self.num_lines = self.num_rest + 1
        self.num_columns = self.num_var + len(self.xf) + len(self.n_xf) + len(self.a) + 1
        self.algorithms = []
        self.generate_matrix()
        self.define_initial_algorithm(matrix_problem[:, 1:])

    # 1 --> < || <=
    # 2 --> > || >=
    # 3 --> =
    def read_config_column(self, line) -> None:
        self.fo_min = line[0] == 1
        for i, e in enumerate(line[1:], 1):
            if e == 0:
                break
            elif e == 1:
                self.xf.append(i)
            elif e == 2:
                self.n_xf.append(i)
                self.a.append(i)
            elif e == 3:
                self.a.append(i)
            else:
                raise Exception("Invalid element in config line")

    def generate_matrix(self) -> None:
        self.algorithms.append(np.zeros([self.num_lines, self.num_columns]))

    def get_algorithm(self, index=-1):
        return self.algorithms[index]

    def get_copy(self):
        return np.copy(self.get_algorithm())

    def define_fo_line(self, line, matrix) -> None:
        matrix[0][:self.num_var] = line[:-1]
        if len(self.a) == 0:
            matrix[0] = np.negative(matrix[0])
        matrix[0][-1] = line[-1]
        for i in range(0, len(self.a)):
            self.a_value = max(x for x in matrix[0][:self.num_var]) * 10000
            matrix[0][self.num_var + len(self.xf) + len(self.n_xf) + i] = self.a_value * (1 if self.fo_min else -1)

    def define_initial_algorithm(self, matrix_problem) -> None:
        matrix = self.get_algorithm()
        self.define_fo_line(matrix_problem[0, :], matrix)

        for i, line in enumerate(matrix[1:], 1):
            line[:self.num_var] = matrix_problem[i][:-1]
            line[-1] = matrix_problem[i][-1]
            matrix[i] = line

        control_column = [0, 0]
        for i, line in enumerate(matrix[1:, self.num_var:-1], 1):
            if i in self.xf:
                line[control_column[0]] = 1
                control_column[0] += 1
            elif i in self.n_xf:
                line[control_column[0]] = -1
                control_column[0] += 1
            if i in self.a:
                line[len(self.n_xf) + len(self.xf) + control_column[1]] = 1
                control_column[1] += 1
            matrix[i, self.num_var:-1] = line

    def execute(self) -> None:
        if len(self.a) > 0:
            self.zero_fo_a()
        self.zero_fo_vars()

    def zero_fo_a(self) -> None:
        matrix = self.get_copy()
        m_line = 0
        for i in self.a:
            if self.fo_min:
                m_line += matrix[i] * self.a_value

            else:
                m_line -= matrix[i] * self.a_value
        m_line -= matrix[0]
        matrix[0] = m_line
        self.algorithms.append(matrix)

    def zero_fo_vars(self) -> None:
        i = 10000000
        while i > 0:
            matrix = self.get_algorithm()
            if self.fo_min:
                if not np.any(matrix[0][:-1] > 0):
                    break
                pivot_var_fo = max(matrix[0][:-1])
            else:
                if not np.any(matrix[0][:-1] < 0):
                    break
                pivot_var_fo = min(matrix[0][:-1])

            pivot_column_index = np.argmax(matrix[0] == pivot_var_fo)
            pivot_line_index = self.define_pivot_line(pivot_column_index, matrix)
            self.generate_new_algorithm(pivot_line_index, pivot_column_index)
            i -= 1

    def define_pivot_line(self, pivot_column_index, matrix) -> int:
        pivot_div_val = 0
        pivot_line_index = 0
        for i in range(1, self.num_lines):
            if matrix[i][pivot_column_index] != 0:
                res = matrix[i][-1] / matrix[i][pivot_column_index]
                if (res < pivot_div_val or pivot_div_val == 0) and res > 0:
                    pivot_div_val = res
                    pivot_line_index = i
        return pivot_line_index

    def generate_new_algorithm(self, pivot_line_index, pivot_column_index) -> None:
        matrix = self.get_copy()
        pivot = matrix[pivot_line_index][pivot_column_index]
        matrix[pivot_line_index] /= pivot
        new_pivot_line = matrix[pivot_line_index]
        for i, line in enumerate(matrix):
            if i == pivot_line_index:
                continue
            line = (-line[pivot_column_index] * new_pivot_line) + line
            matrix[i] = line
        self.algorithms.append(matrix)

    def get_vb(self, index=-1) -> dict:
        vb = {}
        matrix = self.get_algorithm(index)
        for line in matrix:
            for i, v in enumerate(line):
                if v == 1:
                    if i < self.num_var:
                        key = f'x{i}'
                    elif i < self.num_var + len(self.xf) + len(self.n_xf):
                        key = f'xf{i - self.num_var}'
                    else:
                        key = f'a{i-self.num_var - len(self.xf) - len(self.n_xf)}'
                    vb[key] = line[-1]
        vb = OrderedDict(sorted(vb.items()))
        return vb

    def get_header(self) -> list:
        header = []
        for i in range(self.num_var):
            header.append(f'x{i}')
        for i in range(len(self.xf) + len(self.n_xf)):
            header.append(f'xf{i}')
        for i in range(len(self.a)):
            header.append(f'a{i}')
        header.append('b')
        return header

    def get_z(self, index=-1) -> float:
        return self.get_algorithm(index)[0][-1]

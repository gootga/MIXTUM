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

from pathlib import Path
import numpy as np
from time import time
from multiprocessing import Process, Array, Event
from math import ceil, isclose

from PySide6.QtCore import QObject, Signal, Slot



event = Event()

# Compute frequencies given list of alleles
def allele_frequency(alleles):
    freq = 0
    num_alleles = 0

    for a in alleles:
        if a != 9:
            freq += (2 - a) / 2
            num_alleles += 1

    if num_alleles == 0:
        return -1

    return freq / num_alleles

# Compute frequencies of a population
def population_allele_frequencies(file_path, pop_indices, allele_freqs):
    with file_path.open(mode = 'r', encoding = 'utf-8') as file:
        for index, row in enumerate(file):
            allele_freqs[index] = allele_frequency([int(row[i]) for i in pop_indices])
            if event.is_set():
                break



class Core(QObject):
    # Signals
    file_path_set = Signal(str, str)

    input_file_paths_state = Signal(bool)
    pops_file_path_state = Signal(bool)

    geno_file_error = Signal()
    ind_file_error = Signal(int, int)
    snp_file_error = Signal(int, int)

    parsed_pops_error = Signal(list)

    def __init__(self):
        QObject.__init__(self)

        self.geno_file_path = Path('')
        self.ind_file_path = Path('')
        self.snp_file_path = Path('')
        self.pops_file_path = Path('')

        self.num_geno_rows = 0
        self.num_geno_cols = []
        self.num_ind_rows = 0
        self.num_snp_rows = 0

        self.avail_pops = []
        self.avail_pops_indices = {}
        self.snp_names = []
        self.parsed_pops = []
        self.selected_pops = []

        self.num_procs = 1
        self.num_alleles = 0
        self.num_valid_alleles = 0
        self.allele_frequencies = {}

        self.hybrid_pop = ''
        self.parent1_pop = ''
        self.parent2_pop = ''
        self.aux_pops = []
        self.aux_pops_computed = []

        self.alpha_pre_jl = 0
        self.cosine_pre_jl = 0
        self.angle_pre_jl = 0
        self.percentage_pre_jl = 0
        self.f3_test = 0
        self.f4ab_prime = []
        self.f4xb_prime = []
        self.alpha = 0
        self.alpha_error = 0
        self.f4ab_std = []
        self.f4xb_std = []
        self.alpha_std = 0
        self.alpha_std_error =0
        self.cosine_post_jl = 0
        self.angle_post_jl = 0
        self.percentage_post_jl = 0
        self.alpha_ratio = []
        self.alpha_ratio_avg = 0
        self.alpha_ratio_std_dev = 0
        self.alpha_ratio_hist = None
        self.alpha_ratio_hist_bins = 20
        self.num_cases = 0

        self.pca_pops = []
        self.principal_components = []
        self.explained_variance = []
        self.pca_eigenvalues = []

    def check_file_paths(self):
        self.input_file_paths_state.emit(bool(self.geno_file_path.is_file() and self.ind_file_path.is_file() and self.snp_file_path.is_file()))

    @Slot(str)
    def set_geno_file_path(self, file_path):
        self.geno_file_path = Path(file_path)
        self.file_path_set.emit('geno', file_path)
        self.check_file_paths()

    @Slot(str)
    def set_ind_file_path(self, file_path):
        self.ind_file_path = Path(file_path)
        self.file_path_set.emit('ind', file_path)
        self.check_file_paths()

    @Slot(str)
    def set_snp_file_path(self, file_path):
        self.snp_file_path = Path(file_path)
        self.file_path_set.emit('snp', file_path)
        self.check_file_paths()

    @Slot(str)
    def set_pops_file_path(self, file_path):
        self.pops_file_path = Path(file_path)
        self.file_path_set.emit('pops', file_path)
        self.pops_file_path_state.emit(self.pops_file_path.is_file())

    # Count number of rows and columns in .geno input file
    def geno_table_shape(self, progress_callback):
        self.num_geno_rows = 0
        self.num_geno_cols = []

        with self.geno_file_path.open(mode = 'r', encoding = 'utf-8') as file:
            for row in file:
                row = row.rstrip()
                self.num_geno_cols.append(len(row))
                if self.num_geno_rows % 1000 == 0:
                    progress_callback[str, str].emit('geno', f'Number of rows: {self.num_geno_rows}')
                self.num_geno_rows += 1

        self.num_alleles = self.num_geno_rows

        progress_callback[str, str].emit('geno', f'Number of rows: {self.num_geno_rows}')

        return True

    # Parse .ind file containing population indices, and count number of rows
    def parse_ind_file(self, progress_callback):
        self.avail_pops_indices = {}
        self.num_ind_rows = 0

        with self.ind_file_path.open(mode = 'r', encoding = 'utf-8') as file:
            for index, row in enumerate(file):
                progress_callback[str, str].emit('ind', f'Number of rows: {self.num_ind_rows}')

                columns = row.split()
                pop_name = columns[-1]

                if pop_name in self.avail_pops_indices:
                    self.avail_pops_indices[pop_name].append(index)
                else:
                    self.avail_pops_indices[pop_name] = [index]

                self.num_ind_rows += 1

        self.avail_pops = list(self.avail_pops_indices.keys())

        progress_callback[str, str].emit('ind', f'Number of rows: {self.num_ind_rows}')

        return True

    # Parse .snp file containing allele names, and count number of rows
    def parse_snp_file(self, progress_callback):
        self.snp_names = []
        self.num_snp_rows = 0

        with self.snp_file_path.open(mode = 'r', encoding = 'utf-8') as file:
            for row in file:
                columns = row.split()
                self.snp_names.append(columns[0])

                if self.num_snp_rows % 1000 == 0:
                    progress_callback[str, str].emit('snp', f'Number of rows: {self.num_snp_rows}')
                self.num_snp_rows += 1

        progress_callback[str, str].emit('snp', f'Number of rows: {self.num_snp_rows}')

        return True

    # Check input file consistency
    def check_input_files(self):
        valid = True

        if not all(nc == self.num_geno_cols[0] for nc in self.num_geno_cols):
            self.geno_file_error.emit()
            valid = False
        if self.num_ind_rows != self.num_geno_cols[0]:
            self.ind_file_error.emit(self.num_ind_rows, self.num_geno_cols[0])
            valid = False
        if self.num_snp_rows != self.num_geno_rows:
            self.snp_file_error.emit(self.num_snp_rows, self.num_geno_rows)
            valid = False

        return valid

    # Parse input file containing selected populations
    def parse_selected_populations(self, progress_callback):
        self.parsed_pops = []
        num_pops = 0

        with self.pops_file_path.open(mode = 'r', encoding = 'utf-8') as file:
            for row in file:
                columns = row.split()
                self.parsed_pops.append(columns[0])

                num_pops += 1
                progress_callback[str, str].emit('pops', f'Number of pops: {num_pops}')

        return True

    def check_parsed_pops(self):
        if len(self.parsed_pops) > 0 and len(self.avail_pops) > 0:
            missing_pops = [pop for pop in self.parsed_pops if pop not in self.avail_pops]
            self.parsed_pops = [pop for pop in self.parsed_pops if pop in self.avail_pops]
            self.reset_pops()
            if len(missing_pops) > 0:
                self.parsed_pops_error.emit(missing_pops)


    def append_pops(self, pops):
        new_pops = [pop for pop in pops if pop not in self.selected_pops]
        self.selected_pops += new_pops

    def remove_pops(self, pops):
        self.selected_pops = [pop for pop in self.selected_pops if pop not in pops]

    def reset_pops(self):
        self.selected_pops = [pop for pop in self.parsed_pops]

    def time_format(self, seconds):
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f'{minutes} minutes, {seconds:.1f} seconds'

    @Slot()
    def stop_computation(self):
        event.set()

    @Slot(int)
    def set_num_procs(self, np):
        self.num_procs = np

    # Parallel compute frequencies of all populations
    def parallel_compute_populations_frequencies(self, progress_callback):
        event.clear()

        pop_indices = [self.avail_pops_indices[pop] for pop in self.selected_pops]

        num_sel_pops = len(self.selected_pops)
        batch_size = ceil(num_sel_pops / self.num_procs)
        index = 0

        allele_freqs = [Array('d', self.num_alleles) for i in range(num_sel_pops)]

        progress_callback[int].emit(0)
        progress_callback[str, str, int].emit('main', f'Computing {self.num_alleles} frequencies per population for {num_sel_pops} populations in {batch_size} batches of {self.num_procs} parallel processes...', 0)

        num_indices = 0
        t1 = time()

        for batch in range(batch_size):
            procs = []
            computing_pops = []

            for proc in range(self.num_procs):
                if index < num_sel_pops:
                    p = Process(target = population_allele_frequencies, args = (self.geno_file_path, pop_indices[index], allele_freqs[index]))
                    procs.append(p)
                    p.start()
                    computing_pops.append(self.selected_pops[index])
                    num_indices += len(pop_indices[index])
                    index += 1
                else:
                    break

            progress_callback[str, str, int].emit('progress', 'Computing populations: ' + ' '.join(computing_pops), 0)

            for p in procs:
                p.join()

            if event.is_set():
                break

            t2 = time()
            elapsed_time = t2 - t1
            elapsed_time_per_index = elapsed_time / num_indices

            num_remaining_indices = sum([len(indices) for indices in pop_indices[index:]])
            estimated_remaining_time = num_remaining_indices * elapsed_time_per_index

            progress_callback[int].emit(index)
            progress_callback[str, str, int].emit('timing', f'Estimated remaining time: {self.time_format(estimated_remaining_time)}', 0)
            progress_callback[str, str, int].emit('timing', f'Elapsed time: {self.time_format(elapsed_time)}', 1)

        if event.is_set():
            progress_callback[str, str, int].emit('main', 'Computation stopped!', 0)
            progress_callback[str, str, int].emit('progress', 'Allele frequencies unchanged from previous computation.', 0)
            progress_callback[str, str, int].emit('timing', '', 0)
            return False

        progress_callback[str, str, int].emit('main', 'Computation finished.', 0)
        progress_callback[str, str, int].emit('progress', '', 0)
        progress_callback[str, str, int].emit('check', 'Checking and removing invalid SNPs...', 0)

        invalid_indices = np.unique(np.array([index for freqs in allele_freqs for index, freq in enumerate(freqs.get_obj()) if freq == -1], dtype = int))
        self.num_valid_alleles = self.num_alleles - invalid_indices.size

        valid_allele_freqs = []
        for i in range(len(allele_freqs)):
            valid_allele_freqs.append(np.delete(allele_freqs[i], invalid_indices))

        progress_callback[str, str, int].emit('check', 'Checking SNPs finished.', 0)
        progress_callback[str, str, int].emit('check', f'Number of excluded SNPs: {len(invalid_indices)}', 1)

        self.allele_frequencies = {}
        for index, pop in enumerate(self.selected_pops):
            self.allele_frequencies[pop] = np.array(valid_allele_freqs[index], dtype='d')

        self.init_admixture_model()

        return True

    # Init default admixture model
    def init_admixture_model(self):
        self.hybrid_pop = self.selected_pops[0]
        self.parent1_pop = self.selected_pops[1]
        self.parent2_pop = self.selected_pops[2]
        self.aux_pops = self.selected_pops[3:]

    # Set hybrid population and check the rest
    def set_hybrid_pop(self, pop):
        if self.parent1_pop == pop:
            self.parent1_pop = self.hybrid_pop
        elif self.parent2_pop == pop:
            self.parent2_pop = self.hybrid_pop
        elif pop in self.aux_pops:
            self.aux_pops.remove(pop)

        self.hybrid_pop = pop

    # Set parent 1 population and check the rest
    def set_parent1_pop(self, pop):
        if self.hybrid_pop == pop:
            self.hybrid_pop = self.parent1_pop
        elif self.parent2_pop == pop:
            self.parent2_pop = self.parent1_pop
        elif pop in self.aux_pops:
            self.aux_pops.remove(pop)

        self.parent1_pop = pop

    # Set parent 2 population and check the rest
    def set_parent2_pop(self, pop):
        if self.hybrid_pop == pop:
            self.hybrid_pop = self.parent2_pop
        elif self.parent1_pop == pop:
            self.parent1_pop = self.parent2_pop
        elif pop in self.aux_pops:
            self.aux_pops.remove(pop)

        self.parent2_pop = pop

    # Set auxiliary populations
    def set_aux_pops(self, pops):
        self.aux_pops = [pop for pop in pops if pop != self.hybrid_pop and pop != self.parent1_pop and pop != self.parent2_pop]

    # Computation of alpha pre JL
    def mixing_coefficient_pre_jl(self):
        parent_diff = self.allele_frequencies[self.parent1_pop] - self.allele_frequencies[self.parent2_pop]
        self.alpha_pre_jl = np.dot(self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop], parent_diff) / np.dot(parent_diff, parent_diff)

    # Computation of admixture angle pre JL
    def admixture_angle_pre_jl(self):
        xa = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent1_pop]
        xb = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]

        self.cosine_pre_jl = np.dot(xa, xb) / np.sqrt(np.dot(xa, xa) * np.dot(xb, xb))
        self.angle_pre_jl = np.arccos(self.cosine_pre_jl) * 180 / np.pi
        self.percentage_pre_jl = np.arccos(self.cosine_pre_jl) / np.pi

    # Computation of f3
    def f3(self):
        num_alleles = self.allele_frequencies[self.hybrid_pop].size
        self.f3_test = np.dot(self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent1_pop], self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]) / num_alleles

    # Computation of f4 prime
    def f4_prime(self):
        num_aux_pops = len(self.aux_pops)
        num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

        self.f4ab_prime = np.zeros(num_pairs)
        self.f4xb_prime = np.zeros(num_pairs)

        ab = self.allele_frequencies[self.parent1_pop] - self.allele_frequencies[self.parent2_pop]
        xb = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]

        index = 0

        for i in range(num_aux_pops):
            for j in range(i + 1, num_aux_pops):
                ij = self.allele_frequencies[self.aux_pops[i]] - self.allele_frequencies[self.aux_pops[j]]
                norm_ij = np.linalg.norm(ij)
                self.f4ab_prime[index] = np.dot(ab, ij) / norm_ij
                self.f4xb_prime[index] = np.dot(xb, ij) / norm_ij
                index += 1

    # Computation of f4 standard
    def f4_std(self):
        num_aux_pops = len(self.aux_pops)
        num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

        self.f4ab_std = np.zeros(num_pairs)
        self.f4xb_std = np.zeros(num_pairs)

        ab = self.allele_frequencies[self.parent1_pop] - self.allele_frequencies[self.parent2_pop]
        xb = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]

        index = 0

        for i in range(num_aux_pops):
            for j in range(i + 1, num_aux_pops):
                ij = self.allele_frequencies[self.aux_pops[i]] - self.allele_frequencies[self.aux_pops[j]]
                self.f4ab_std[index] = np.dot(ab, ij)
                self.f4xb_std[index] = np.dot(xb, ij)
                index += 1

        num_alleles = self.allele_frequencies[self.hybrid_pop].size

        self.f4ab_std /= num_alleles
        self.f4xb_std /= num_alleles

    def get_aux_pop_pair(self, index):
        num_aux_pops = len(self.aux_pops_computed)
        k = 0
        for i in range(num_aux_pops):
            for j in range(i + 1, num_aux_pops):
                if k == index:
                    return self.aux_pops_computed[i], self.aux_pops_computed[j]
                k += 1
        return '', ''


    # Least squares fit
    def least_squares(self, x, y):
        dim = len(x)

        A = np.vstack([x, np.zeros(dim)]).T
        alpha = np.linalg.lstsq(A, y)[0][0]

        Q = 0
        for i in range(dim):
            Q += (y[i] - alpha * x[i]) ** 2

        x_avg = 0
        for i in range(dim):
            x_avg += x[i]
        x_avg /= dim

        x_dev = 0
        for i in range(dim):
            x_dev += (x[i] - x_avg) ** 2

        s_alpha = np.sqrt(Q / ((dim - 2) * x_dev))
        t = 1.98

        error = s_alpha * t

        return alpha, error

    # Computation of alpha
    def alpha_prime(self):
        self.alpha, self.alpha_error = self.least_squares(self.f4ab_prime, self.f4xb_prime)

    # Computation of alpha standard
    def alpha_standard(self):
        self.alpha_std, self.alpha_std_error = self.least_squares(self.f4ab_std, self.f4xb_std)

    # Computation of admixture angle post JL
    def admixture_angle_post_jl(self):
        num_aux_pops = len(self.aux_pops)

        xa = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent1_pop]
        xb = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(num_aux_pops):
            for j in range(i + 1, num_aux_pops):
                ij = self.allele_frequencies[self.aux_pops[i]] - self.allele_frequencies[self.aux_pops[j]]

                xaij = np.dot(xa, ij)
                xbij = np.dot(xb, ij)
                ijij = np.dot(ij, ij)

                sum1 += xaij * xbij / ijij
                sum2 += (xaij ** 2) / ijij
                sum3 += (xbij ** 2) / ijij

        self.cosine_post_jl = sum1 / np.sqrt(sum2 * sum3)
        self.angle_post_jl = np.arccos(self.cosine_post_jl) * 180 / np.pi
        self.percentage_post_jl = np.arccos(self.cosine_post_jl) / np.pi

    # Computation of f4-ratio
    def f4_ratio(self):
        num_aux_pops = len(self.aux_pops)
        num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

        xb = self.allele_frequencies[self.hybrid_pop] - self.allele_frequencies[self.parent2_pop]
        ab = self.allele_frequencies[self.parent1_pop] - self.allele_frequencies[self.parent2_pop]

        self.alpha_ratio = np.zeros(num_pairs)

        index = 0

        for i in range(num_aux_pops):
            for j in range(i + 1, num_aux_pops):
                ij = self.allele_frequencies[self.aux_pops[i]] - self.allele_frequencies[self.aux_pops[j]]
                if not isclose(np.dot(ab, ij), 0, abs_tol=1e-15):
                    self.alpha_ratio[index] = np.dot(xb, ij) / np.dot(ab, ij)
                index += 1

        alpha_01 = self.alpha_ratio[(self.alpha_ratio >= 0) & (self.alpha_ratio <= 1)]
        self.alpha_ratio_avg = np.average(alpha_01)
        self.alpha_ratio_std_dev = np.std(alpha_01, dtype='d') * 1.96
        self.alpha_ratio_hist = np.histogram(self.alpha_ratio, self.alpha_ratio_hist_bins)
        self.num_cases = alpha_01.size

    # Computation of f4-ratio histogram
    def compute_f4_ratio_histogram(self, bins):
        self.alpha_ratio_hist_bins = bins
        self.alpha_ratio_hist = np.histogram(self.alpha_ratio, self.alpha_ratio_hist_bins)

    # PCA of allele frequencies
    def compute_pca(self, pops):
        # frequencies = np.array([(1 - self.allele_frequencies[pop]) * 2 for pop in pops], dtype='d')
        frequencies = np.array([self.allele_frequencies[pop] for pop in pops], dtype='d')
        centers = np.mean(frequencies, axis=0, dtype='d')
        a = np.array([freqs - centers for freqs in frequencies], dtype='d')
        aat = np.matmul(a, np.transpose(a))
        self.pca_eigenvalues, eigenvectors = np.linalg.eigh(aat / (aat.shape[0] - 1))
        w = np.einsum('ji,jk', a, eigenvectors[:, ::-1])
        norms = np.linalg.norm(w, axis=0)
        wn = w / norms
        self.principal_components = np.einsum('ij,jk', a, wn)
        self.explained_variance = 100 * np.flip(self.pca_eigenvalues)[:3]/np.sum(self.pca_eigenvalues)
        self.pca_pops = pops

    # Compute all results
    def compute_results(self, progress_callback):
        progress_callback[int].emit(0)
        self.mixing_coefficient_pre_jl()
        progress_callback[int].emit(1)
        self.admixture_angle_pre_jl()
        progress_callback[int].emit(2)
        self.f3()
        progress_callback[int].emit(3)
        self.f4_prime()
        progress_callback[int].emit(4)
        self.alpha_prime()
        progress_callback[int].emit(5)
        self.f4_std()
        progress_callback[int].emit(6)
        self.alpha_standard()
        progress_callback[int].emit(7)
        self.admixture_angle_post_jl()
        progress_callback[int].emit(8)
        self.f4_ratio()
        progress_callback[int].emit(9)

        self.aux_pops_computed = self.aux_pops

        return True

    # Get result data in text form
    def admixture_data(self):
        num_aux_pops = len(self.aux_pops)
        num_aux_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

        text = f'Admixture model: {self.hybrid_pop} = {self.parent1_pop} + {self.parent2_pop}\n'
        text += f'SNPs employed: {self.num_valid_alleles} / {self.num_alleles}\n'
        text += f'Auxiliary populations: {num_aux_pops}\n'
        text += f'Auxiliary pairs: {num_aux_pairs}\n'
        text += f'Cos pre-JL:  {self.cosine_pre_jl:7.4f} ---> Angle pre-JL:  {self.angle_pre_jl:7.2f} deg vs 180 deg: {self.percentage_pre_jl:.1%}\n'
        text += f'Cos post-JL: {self.cosine_post_jl:7.4f} ---> Angle post-JL: {self.angle_post_jl:7.2f} deg vs 180 deg: {self.percentage_post_jl:.1%}\n'
        text += f'Alpha pre-JL:     {self.alpha_pre_jl:6.4f}\n'
        text += f'Alpha post-JL:    {self.alpha:6.4f} +/- {self.alpha_error:6.4f} (95% CI) (f4-prime, renormalized)\n'
        text += f'Alpha NR post-JL: {self.alpha_std:6.4f} +/- {self.alpha_std_error:6.4f} (95% CI) (f4, standard)\n'
        text += f'f4-ratio average if [0, 1]: {self.alpha_ratio_avg:6.4f} +/- {self.alpha_ratio_std_dev:6.4f} (95% CI), {self.num_cases} cases\n'
        text += f'Standard admixture test: f3(parent1, parent2; hybrid) < 0 ? {self.f3_test:8.6f}'

        return text

    # Save frequencies
    def save_population_allele_frequencies(self, file_path_str):
        file_path = Path(file_path_str)

        with file_path.open(mode='w', encoding='utf-8') as file:
            pops_width = max([len(name) for name in self.selected_pops])
            prec = 6
            col_width = max(prec + 7, pops_width)

            headers_format = ' '.join([f'{{{i}:^{col_width}}}' for i, pop in enumerate(self.selected_pops)])
            headers = headers_format.format(*self.selected_pops)
            file.write(headers + '\n')

            row_format = ' '.join([f'{{{i}: {col_width}.{prec}E}}' for i, pop in enumerate(self.selected_pops)])

            for allele_index in range(self.num_valid_alleles):
                row = [freqs[allele_index] for pop, freqs in self.allele_frequencies.items()]
                file.write(row_format.format(*row) + '\n')

    # Save f4 points
    def save_f4_points(self, file_path_str):
        file_path = Path(file_path_str)

        with file_path.open(mode = 'w', encoding = 'utf-8') as file:
            aux_pops_width = max([len(name) for name in self.aux_pops_computed])
            prec = 6
            col_width = prec + 7

            headers = '{0:^{col_width}} {1:^{col_width}} {2:^{col_width}} {3:^{col_width}} {4:^{col_width}} {5:^{aux_pops_width}} {6:^{aux_pops_width}}'.format('f4primeAB', 'f4primeXB', 'f4AB', 'f4XB', 'f4-ratio', 'Aux1', 'Aux2', col_width = col_width, aux_pops_width = aux_pops_width)
            file.write(headers + '\n')

            num_aux_pops = len(self.aux_pops)

            index = 0

            for i in range(num_aux_pops):
                for j in range(i + 1, num_aux_pops):
                    row = '{0: {col_width}.{prec}E} {1: {col_width}.{prec}E} {2: {col_width}.{prec}E} {3: {col_width}.{prec}E} {4: {col_width}.{prec}E} {5:{aux_pops_width}} {6:{aux_pops_width}}'.format(self.f4ab_prime[index], self.f4xb_prime[index], self.f4ab_std[index], self.f4xb_std[index], self.alpha_ratio[index], self.aux_pops[i], self.aux_pops[j], prec = prec, col_width = col_width, aux_pops_width = aux_pops_width)
                    file.write(row + '\n')
                    index += 1

    # Save results
    def save_admixture_data(self, file_path_str):
        file_path = Path(file_path_str)

        with file_path.open(mode = 'w', encoding = 'utf-8') as file:
            num_aux_pops = len(self.aux_pops_computed)
            num_aux_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

            file.write(f'Admixture model: {self.hybrid_pop} = {self.parent1_pop} + {self.parent2_pop}\n')
            file.write(f'SNPs employed: {self.num_valid_alleles} / {self.num_alleles}\n')
            file.write(f'Auxiliary populations: {num_aux_pops}\n')
            file.write(f'Auxiliary pairs: {num_aux_pairs}\n')
            file.write(f'Cos pre-JL:  {self.cosine_pre_jl:7.4f} ---> Angle pre-JL:  {self.angle_pre_jl:7.2f} deg vs 180 deg: {self.percentage_pre_jl:.1%}\n')
            file.write(f'Cos post-JL: {self.cosine_post_jl:7.4f} ---> Angle post-JL: {self.angle_post_jl:7.2f} deg vs 180 deg: {self.percentage_post_jl:.1%}\n')
            file.write(f'Alpha pre-JL:     {self.alpha_pre_jl:6.4f}\n')
            file.write(f'Alpha post-JL:    {self.alpha:6.4f} +/- {self.alpha_error:6.4f} (95% CI) (f4-prime, renormalized)\n')
            file.write(f'Alpha NR post-JL: {self.alpha_std:6.4f} +/- {self.alpha_std_error:6.4f} (95% CI) (f4, standard)\n')
            file.write(f'f4-ratio average if [0, 1]: {self.alpha_ratio_avg:6.4f} +/- {self.alpha_ratio_std_dev:6.4f} (95% CI), {self.num_cases} cases\n')
            file.write(f'Standard admixture test: f3(parent1, parent2; hybrid) < 0 ? {self.f3_test:8.6f}\n')
            file.write('Auxiliary population names:\n')
            file.write('\n'.join(self.aux_pops_computed))

    # Save PCA data
    def save_pca_data(self, file_path_str):
        file_path = Path(file_path_str)

        prec = 6
        col_width = prec + 7
        row_format = ' '.join([f'{{{i}: {col_width}.{prec}E}}' for i in range(6)])
        eig_format = ' '.join([f'{{{i}: {col_width}.{prec}E}}' for i in range(len(self.pca_eigenvalues))])

        pops_width = max(max([len(name) for name in self.pca_pops]), len('Populations'))
        headers = '{0:^{pops_width}} {1:^{col_width}} {2:^{col_width}} {3:^{col_width}} {4:^{col_width}}'.format('Populations', 'PC1', 'PC2', 'PC3', '...', col_width=col_width, pops_width=pops_width)

        with file_path.open(mode='w', encoding='utf-8') as file:
            file.write(headers + '\n')
            for i, pc in enumerate(self.principal_components):
                pop_name = '{0:^{pops_width}}'.format(self.pca_pops[i], pops_width=pops_width)
                file.write(pop_name + row_format.format(*pc[:6]) + '\n')
            file.write('\nPC eigenvalues\n')
            file.write(eig_format.format(*np.flip(self.pca_eigenvalues)) + '\n')
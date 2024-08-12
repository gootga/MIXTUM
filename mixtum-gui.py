from pathlib import Path
from collections import defaultdict
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import panel as pn
import io
import zipfile
from datetime import datetime



pn.extension()
pn.extension('tabulator')
pn.extension('filedropper')



# Globals

geno_file_path = None
ind_file_path = None
snp_file_path = None
#input_triad = {}

populations_dict = defaultdict(list)
populations = []

parsed_sel_pops = []
sel_pops = []
snp_names = []

num_alleles = 0
allele_frequencies = defaultdict(list)

invalid_indices = []
num_valid_alleles = 0

roles_pops = []

hybrid = ''
parent1 = ''
parent2 = ''
auxiliaries = []

alpha_pre_jl = None

f4ab_prime = None
f4xb_prime = None
alpha = None
alpha_error = None

f4ab_std = None
f4xb_std = None
alpha_std = None
alpha_std_error = None

alpha_ratio = None
alpha_ratio_avg = None
alpha_ratio_std_dev = None
alpha_ratio_hist = None

cosine_pre_jl = None
angle_pre_jl = None
percentage_pre_jl = None

cosine_post_jl = None
angle_post_jl = None
percentage_post_jl = None

num_cases = None

f3_test = None



# Input and output widgets

zip_file_dropper = pn.widgets.FileDropper(accepted_filetypes = ['application/zip'], max_file_size = '1000MB', multiple = False)
dat_file_dropper = pn.widgets.FileDropper(max_file_size = '1MB', multiple = False)
load_files_button = pn.widgets.Button(name = 'Parse and check input files', button_type = 'primary', disabled = False)

alert_pane = pn.pane.Alert('### Input files upload\nPlease, upload a ZIP with a triad of .geno, .ind and .snp input files, and optionally an input file with a list of selected populations.\nThen press the parse button.', alert_type = 'primary')

reset_sel_pops_button = pn.widgets.Button(name = 'Reset selected populations', button_type = 'primary')

compute_freqs_button = pn.widgets.Button(name = 'Compute allele frequencies', button_type = 'primary', disabled = True)
freqs_download_button = pn.widgets.FileDownload(label = 'Download allele frequencies', disabled = True, button_type = 'primary')

hybrid_select = pn.widgets.Select(name = 'Hybrid', options = [], size = 10)
parent1_select = pn.widgets.Select(name = 'Parent 1', options = [], size = 10)
parent2_select = pn.widgets.Select(name = 'Parent 2', options = [], size = 10)
aux_pops_select = pn.widgets.MultiSelect(name = 'Auxiliaries', options = [], size = 10)


plot_width_input = pn.widgets.FloatInput(name = 'Plot width (inches)', value = 4, step = 0.1, start = 0.1)
plot_height_input = pn.widgets.FloatInput(name = 'Plot height (inches)', value = 4, step = 0.1, start = 0.1)
plot_title_size_input = pn.widgets.IntInput(name = 'Plot title font size', value = 10, step = 1, start = 1)
plot_label_size_input = pn.widgets.IntInput(name = 'Plot labels font size', value = 10, step = 1, start = 1)
compute_results_button = pn.widgets.Button(name = 'Compute results', button_type = 'primary', disabled = True)
results_download_button = pn.widgets.FileDownload(label = 'Download results', disabled = True, button_type = 'primary')
f4_points_download_button = pn.widgets.FileDownload(label = 'Download f4-points', disabled = True, button_type = 'primary')
resulting_data_output = pn.pane.Markdown()



# Tables

avail_pops_filter = pn.widgets.TextInput(name = 'Search populations:', placeholder = 'Enter population name', disabled = True)
avail_pops_table = pn.widgets.Tabulator(show_index = False, disabled = True, selectable = False, pagination = 'local', page_size = 10, align = ('start'), widths = {'population': '100%'})
sel_pops_table = pn.widgets.Tabulator(show_index = False, disabled = True, selectable = False, sortable = False, pagination = 'local', page_size = 10, align = ('start'), widths = {'population': '100%'}, buttons = {'remove': "<i class='fa fa-times'></i>"})



# Plot panes

pane_margin = 10

f4prime_fit_pane = pn.pane.Matplotlib(align = ('center', 'start'), margin = pane_margin)
f4_fit_pane = pn.pane.Matplotlib(align = ('center', 'start'), margin = pane_margin)
f4_ratio_histogram_pane = pn.pane.Matplotlib(align = ('center', 'start'), margin = pane_margin)



def set_alert_pane(text, type):
    alert_pane.object = text
    alert_pane.alert_type = type



def reset_alert_pane(event):
    text = '### Input files upload\nPlease, upload a ZIP with triad of .geno, .ind and .snp input files, and optionally an input file with a list of selected populations.\nThen press the parse button.'
    alert_pane.object = text
    alert_pane.alert_type = 'primary'

    if zip_file_dropper.value:
        load_files_button.disabled = False
    else:
        load_files_button.disabled = True



zip_file_dropper.param.watch(reset_alert_pane, 'value')



def set_pops_table_data(pops, table):
    idx = list(range(len(pops)))
    df = pd.DataFrame({'population': pops}, index = idx)
    table.value = df



def contains_filter(df, pattern, column):
    if not pattern:
        return df
    return df[df[column].str.contains(pattern, case = False)]



avail_pops_table.add_filter(pn.bind(contains_filter, pattern = avail_pops_filter, column = 'population'))



def set_sel_pops_data(event):
    global sel_pops

    pop = event.value
    if pop in sel_pops:
        sel_pops.remove(pop)
    else:
        sel_pops.insert(0, pop)

    set_pops_table_data(sel_pops, sel_pops_table)

    if len(sel_pops) > 0:
        compute_freqs_button.disabled = False
    else:
        compute_freqs_button.disabled = True



avail_pops_table.on_click(set_sel_pops_data)



def remove_sel_pop(event):
    if event.column != 'remove':
        return

    global sel_pops
    index = int(event.row)
    sel_pops.pop(index)

    set_pops_table_data(sel_pops, sel_pops_table)

    if len(sel_pops) == 0:
        compute_freqs_button.disabled = True



sel_pops_table.on_click(remove_sel_pop)



def set_admixture_model():
    global hybrid, parent1, parent2, auxiliaries

    hybrid = hybrid_select.value
    parent1 = parent1_select.value
    parent2 = parent2_select.value
    auxiliaries = aux_pops_select.value



def init_selects_options():
    global roles_pops
    roles_pops = [pop for pop in sel_pops]

    hybrid_select.options = roles_pops
    parent1_select.options = roles_pops
    parent2_select.options = roles_pops
    aux_pops_select.options = roles_pops

    hybrid_select.value = roles_pops[0]
    parent1_select.value = roles_pops[1]
    parent2_select.value = roles_pops[2]
    aux_pops_select.value = roles_pops[3:]

    set_admixture_model()



def set_selects_values(event):
    select_name = event.obj.name
    old_pop = event.old
    new_pop = event.new

    if select_name == 'Auxiliaries':
        pops = [hybrid_select.value, parent1_select.value, parent2_select.value]
        aux_pops_select.value = [pop for pop in new_pop if pop not in pops]
    else:
        if new_pop in aux_pops_select.value:
            aux_pops_select.value = [pop for pop in aux_pops_select.value if pop != new_pop]
        else:
            selects = []
            if select_name == 'Hybrid':
                selects = [parent1_select, parent2_select]
            elif select_name == 'Parent 1':
                selects = [hybrid_select, parent2_select]
            elif select_name == 'Parent 2':
                selects = [hybrid_select, parent1_select]

            for sel in selects:
                if sel.value == new_pop and old_pop != None:
                    sel.value = old_pop

    set_admixture_model()

    compute_results_button.disabled = (len(aux_pops_select.value) == 0)



hybrid_select.param.watch(set_selects_values, 'value')
parent1_select.param.watch(set_selects_values, 'value')
parent2_select.param.watch(set_selects_values, 'value')
aux_pops_select.param.watch(set_selects_values, 'value')



def parse_selected_populations(data):
    global parsed_sel_pops
    read_sel_pops = []

    pop_lines = [line.split() for line in data.splitlines() if not line.startswith('#')]
    read_sel_pops = [pop_line[0] for pop_line in pop_lines]

    invalid_pops = []
    for pop in read_sel_pops:
        if not pop in populations:
            invalid_pops.append(pop)

    global sel_pops
    sel_pops = [pop for pop in read_sel_pops if pop not in invalid_pops]

    global parsed_sel_pops
    parsed_sel_pops = [pop for pop in read_sel_pops if pop not in invalid_pops]

    if len(sel_pops) > 0:
        set_pops_table_data(sel_pops, sel_pops_table)
        compute_freqs_button.disabled = False
    else:
        compute_freqs_button.disabled = True



def reset_sel_pops(event):
    global sel_pops
    sel_pops = [pop for pop in parsed_sel_pops]
    set_pops_table_data(sel_pops, sel_pops_table)

    if len(sel_pops) == 0:
        compute_freqs_button.disabled = True
        freqs_download_button.disabled = True



pn.bind(reset_sel_pops, reset_sel_pops_button, watch = True)



def parse_populations():
    global populations_dict, populations

    populations_dict = defaultdict(list)
    populations = []

    num_rows = 0

    text = f'### Parsing and checking {ind_file_path}\n'
    text_lines = [text, '']

    #for index, row in enumerate(input_triad[ind_file_path]):
    for index, row in enumerate(pn.state.cache['.ind']):
        columns = row.split()
        pop_name = columns[-1]
        populations_dict[pop_name].append(index)

        num_rows += 1
        if (num_rows % 1000 == 0):
            text_lines[-1] = f'Number of rows: {num_rows}'
            set_alert_pane('\n'.join(text_lines), 'warning')

    populations = list(populations_dict.keys())

    return num_rows



def parse_snp_names():
    global snp_names
    snp_names = []

    num_alleles = 0

    text = f'### Parsing and checking {snp_file_path}\n'
    text_lines = [text, '']

    #for row in input_triad[snp_file_path]:
    for row in pn.state.cache['.snp']:
        columns = row.split()
        snp_names.append(columns[0])

        num_alleles += 1
        if (num_alleles % 10000 == 0):
            text_lines[-1] = f'Number of alleles: {num_alleles}'
            set_alert_pane('\n'.join(text_lines), 'warning')

    return num_alleles



def geno_table_shape():
    num_rows = 0
    num_columns = []

    text = f'### Parsing and checking {geno_file_path}\n'
    text_lines = [text, '']

    #for row in input_triad[geno_file_path]:
    for row in pn.state.cache['.geno']:
        num_rows += 1
        num_columns.append(len(row))

        if (num_rows % 1000 == 0):
            text_lines[-1] = f'Number of rows: {num_rows}'
            set_alert_pane('\n'.join(text_lines), 'warning')

    text_lines[-1] = f'Number of rows: {num_rows}'
    set_alert_pane('\n'.join(text_lines), 'warning')

    return num_rows, num_columns



def zip_file_names():
    files = []
    with zipfile.ZipFile(io.BytesIO(next(iter(zip_file_dropper.value.items()))[1])) as zip_ref:
        for file in zip_ref.namelist():
            if zipfile.Path(zip_ref, file).is_file():
                files.append(file)
    return files



def unzip_input_file():
    #global input_triad
    #input_triad = {}

    pn.state.cache['.geno'] = None
    pn.state.cache['.ind'] = None
    pn.state.cache['.snp'] = None

    with zipfile.ZipFile(io.BytesIO(next(iter(zip_file_dropper.value.items()))[1])) as zip_ref:
        files = zip_ref.namelist()
        for file in files:
            file_path = Path(file)
            if file_path.suffix in ['.geno', '.ind', '.snp']:
                #input_triad[file] = zip_ref.read(file).decode('utf-8').splitlines()
                pn.state.cache[file_path.suffix] = []
                with zip_ref.open(file) as zip_file:
                    for row in zip_file:
                        pn.state.cache[file_path.suffix].append(row.decode('utf-8').rstrip())
        zip_file_dropper.value = {}



def load_input_files(event):
    file_paths = zip_file_names()

    # Check selected files for invalid suffixes

    invalid_file_suffixes = []

    for fp in file_paths:
        file_path = Path(fp)
        if file_path.suffix not in ['.geno', '.ind', '.snp']:
            invalid_file_suffixes.append(file_path.suffix + '\n')

    if len(invalid_file_suffixes) > 0:
        invalid_suff_list = '- '.join(invalid_file_suffixes)
        text = f'### Unrecognized file suffixes found in ZIP file:\n{invalid_suff_list}\nValid input files end with the suffixes: .geno, .ind and .snp'
        set_alert_pane(text, 'danger')
        return

    # Check for triad

    suffixes = [Path(fp).suffix for fp in file_paths]

    count_geno = suffixes.count('.geno')
    count_ind = suffixes.count('.ind')
    count_snp = suffixes.count('.snp')

    if count_geno != 1 or count_ind != 1 or count_snp != 1:
        text = '### Wrong number of input files\nPlease, upload a triad of input files (.geno, .ind, .snp)'
        set_alert_pane(text, 'danger')
        return

    # Succesful selection

    text = '### Extracting ZIP file to memory\nPlease, wait...'
    set_alert_pane(text, 'warning')

    unzip_input_file()

    global geno_file_path, ind_file_path, snp_file_path

    for fp in file_paths:
        file_path = Path(fp)
        if file_path.suffix == '.geno':
            geno_file_path = fp
        elif file_path.suffix == '.ind':
            ind_file_path = fp
        elif file_path.suffix == '.snp':
            snp_file_path = fp

    # Parsing

    # Check .geno table

    num_geno_rows, num_geno_columns = geno_table_shape()

    if not all(nc == num_geno_columns[0] for nc in num_geno_columns):
        text = f'### Parsing failed\nIn {geno_file_path}: Not all rows are of equal number of columns'
        set_alert_pane(text, 'danger')
        return

    # Parse and check number of alleles and rows

    global num_alleles

    num_alleles = parse_snp_names()

    if num_alleles != num_geno_rows:
        text = f'### Parsing failed\nNumber of alleles ({num_alleles}) in .snp file is not equal to number of rows ({num_geno_rows}) in .geno file'
        set_alert_pane(text, 'danger')
        return

    # Parse and check columns

    num_ind_rows = parse_populations()

    if num_ind_rows != num_geno_columns[0]:
        text = f'### Parsing failed\nNumber of rows ({num_ind_rows}) in .ind file is not equal to number of columns ({num_geno_columns[0]}) in .geno file'
        set_alert_pane(text, 'danger')
        return

    # Parse and check selected populations

    global sel_pops
    global parsed_sel_pops

    dat_file_path = ''

    if dat_file_dropper.value:
        dat_file_data = next(iter(dat_file_dropper.value.items()))[1].decode('utf-8')
        parse_selected_populations(dat_file_data)
        dat_file_path = next(iter(dat_file_dropper.value.items()))[0]
    else:
        sel_pops = []
        parsed_sel_pops = []
        compute_freqs_button.disabled = True

    freqs_download_button.disabled = True

    # Set tables

    set_pops_table_data(populations, avail_pops_table)
    set_pops_table_data(sel_pops, sel_pops_table)

    avail_pops_filter.disabled = False

    text = f'### Parsing successful\nParsed input files seem to have a correct structure:\n- {num_geno_rows} rows and {num_geno_columns[0]} columns in {geno_file_path}\n- {num_alleles} alleles in {snp_file_path}\n- {num_ind_rows} rows and {len(populations_dict)} populations in {ind_file_path}'
    if len(sel_pops) > 0:
        text += f'\n- {len(sel_pops)} selected populations in {dat_file_path}'
    set_alert_pane(text, 'success')



pn.bind(load_input_files, load_files_button, watch = True)



def invalid_allele_indices():
    indices = []

    for pop, freqs in allele_frequencies.items():
        for index, freq in enumerate(freqs):
            if freq == -1:
                indices.append(index)

    global invalid_indices
    invalid_indices = np.unique(np.array(indices, dtype = int))

    global num_valid_alleles
    num_valid_alleles = num_alleles - invalid_indices.size



def remove_invalid_alleles():
    global allele_frequencies
    for pop, freqs in allele_frequencies.items():
        allele_frequencies[pop] = np.delete(freqs, invalid_indices)



def time_format(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f'{minutes} minutes, {seconds} seconds'


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



def population_allele_frequencies(pop_indices, allele_freqs):
    for index, row in enumerate(geno_file_data.splitlines()):
        allele_freqs[index] = allele_frequency([int(row[i]) for i in pop_indices])



def compute_populations_frequencies(event):
    num_pops = len(sel_pops)

    text_lines = ['### Computing']
    text = f'Computing {num_alleles} frequencies per population for {num_pops} populations.'
    text_lines.append(text)
    set_alert_pane('\n'.join(text_lines), 'warning')

    text_lines.append('')

    global allele_frequencies
    allele_frequencies = defaultdict(list)
    for pop in sel_pops:
        allele_frequencies[pop] = np.zeros(num_alleles)

    comp_time = []
    t0 = time()

    #for index, row in enumerate(input_triad[geno_file_path]):
    for index, row in enumerate(pn.state.cache['.geno']):
        if index % 1000 == 0 or index == num_alleles - 1:
            t1 = time()

        for pop in sel_pops:
            allele_frequencies[pop][index] = allele_frequency([int(row[i]) for i in populations_dict[pop]])

        if index % 1000 == 0 or index == num_alleles - 1:
            t2 = time()

            comp_time.append(t2 - t1)

            avg_comp_time = sum(comp_time) / len(comp_time)
            remaining_comps = num_alleles - index - 1
            remaining_time = remaining_comps * avg_comp_time

            percentage_done = (index + 1) / num_alleles

            t3 = time()
            elapsed_time = t3 - t0

            text_lines[-1] = f'Processed: {index + 1} / {num_alleles} rows ({percentage_done:.1%})\nEstimated remaining time: {time_format(remaining_time)}\nElapsed time {time_format(elapsed_time)}'
            set_alert_pane('\n'.join(text_lines), 'warning')

    text_lines[0] = '### Finished'
    set_alert_pane('\n'.join(text_lines), 'success')

    invalid_allele_indices()

    if len(invalid_indices) > 0:
        text = f'Number of excluded SNPs: {len(invalid_indices)}'
        text_lines.append(text)
        set_alert_pane('\n'.join(text_lines), 'success')

        remove_invalid_alleles()

    init_selects_options()

    compute_results_button.disabled = False
    resulting_data_output.object = ''

    f4prime_fit_pane.object = None
    f4_fit_pane.object = None
    f4_ratio_histogram_pane.object = None

    freqs_download_button.disabled = False
    results_download_button.disabled = True
    f4_points_download_button.disabled = True




pn.bind(compute_populations_frequencies, compute_freqs_button, watch = True)



def mixing_coefficient_pre_jl(hybrid_freqs, parent1_freqs, parent2_freqs):
    parent_diff = parent1_freqs - parent2_freqs
    return np.dot(hybrid_freqs - parent2_freqs, parent_diff) / np.dot(parent_diff, parent_diff)



def admixture_angle_pre_jl(hybrid_freqs, parent1_freqs, parent2_freqs):
    xa = hybrid_freqs - parent1_freqs
    xb = hybrid_freqs - parent2_freqs

    cosine = np.dot(xa, xb) / np.sqrt(np.dot(xa, xa) * np.dot(xb, xb))
    angle = np.arccos(cosine)
    percentage = angle / np.pi

    return cosine, angle * 180 / np.pi, percentage



def f3(hybrid_freqs, parent1_freqs, parent2_freqs):
    num_alleles = hybrid_freqs.size
    return np.dot(hybrid_freqs - parent1_freqs, hybrid_freqs - parent2_freqs) / num_alleles



def f4_prime(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs):
    num_aux_pops = len(aux_freqs)
    num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    f4ab_prime = np.zeros(num_pairs)
    f4xb_prime = np.zeros(num_pairs)

    ab = parent1_freqs - parent2_freqs
    xb = hybrid_freqs - parent2_freqs

    index = 0

    for i in range(num_aux_pops):
        for j in range(i + 1, num_aux_pops):
            ij = aux_freqs[i] - aux_freqs[j]
            norm_ij = np.linalg.norm(ij)
            f4ab_prime[index] = np.dot(ab, ij) / norm_ij
            f4xb_prime[index] = np.dot(xb, ij) / norm_ij
            index += 1

    return f4ab_prime, f4xb_prime



def f4_std(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs):
    num_aux_pops = len(aux_freqs)
    num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    f4ab_std = np.zeros(num_pairs)
    f4xb_std = np.zeros(num_pairs)

    ab = parent1_freqs - parent2_freqs
    xb = hybrid_freqs - parent2_freqs

    index = 0

    for i in range(num_aux_pops):
        for j in range(i + 1, num_aux_pops):
            ij = aux_freqs[i] - aux_freqs[j]
            f4ab_std[index] = np.dot(ab, ij)
            f4xb_std[index] = np.dot(xb, ij)
            index += 1

    num_alleles = hybrid_freqs.size

    return f4ab_std / num_alleles, f4xb_std / num_alleles



def least_squares(x, y):
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



def admixture_angle_post_jl(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs):
    num_aux_pops = len(aux_freqs)
    num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    xa = hybrid_freqs - parent1_freqs
    xb = hybrid_freqs - parent2_freqs

    sum1 = 0
    sum2 = 0
    sum3 = 0


    for i in range(num_aux_pops):
        for j in range(i + 1, num_aux_pops):
            ij = aux_freqs[i] - aux_freqs[j]

            xaij = np.dot(xa, ij)
            xbij = np.dot(xb, ij)
            ijij = np.dot(ij, ij)

            sum1 += xaij * xbij / ijij
            sum2 += (xaij ** 2) / ijij
            sum3 += (xbij ** 2) / ijij

    cosine = sum1 / np.sqrt(sum2 * sum3)
    angle = np.arccos(cosine)
    percentage = angle / np.pi

    return cosine, angle * 180 / np.pi, percentage



def f4_ratio(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs):
    num_aux_pops = len(aux_freqs)
    num_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    xb = hybrid_freqs - parent2_freqs
    ab = parent1_freqs - parent2_freqs

    alpha = np.zeros(num_pairs)

    index = 0

    for i in range(num_aux_pops):
        for j in range(i + 1, num_aux_pops):
            ij = aux_freqs[i] - aux_freqs[j]
            alpha[index] = np.dot(xb, ij) / np.dot(ab, ij)
            index += 1

    alpha_01 = alpha[(alpha >= 0) & (alpha <= 1)]
    alpha_avg = np.average(alpha_01)
    alpha_std_dev = np.std(alpha_01) * 1.96
    alpha_hist = np.histogram(alpha, 20)

    return alpha, alpha_avg, alpha_std_dev, alpha_hist, alpha_01.size



def plot_fit(x, y, alpha, title, xlabel, ylabel):
    fig = Figure(figsize = (plot_width_input.value, plot_height_input.value), layout = 'constrained')
    ax = fig.subplots()

    ax.set_title(title, fontsize = plot_title_size_input.value)

    ax.set_xlabel(xlabel, fontsize = plot_label_size_input.value)
    ax.set_ylabel(ylabel, fontsize = plot_label_size_input.value)

    ax.plot(x, y, '.')
    ax.plot(x, alpha * x)

    return fig



def plot_histogram(histogram, title, xlabel, ylabel):
    counts = histogram[0]
    edges = histogram[1]

    fig = Figure(figsize = (plot_width_input.value, plot_height_input.value), layout = 'constrained')
    ax = fig.subplots()

    ax.set_title(title, fontsize = plot_title_size_input.value)

    ax.set_xlabel(xlabel, fontsize = plot_label_size_input.value)
    ax.set_ylabel(ylabel, fontsize = plot_label_size_input.value)

    ax.bar(edges[:-1], counts, width = np.diff(edges), edgecolor = 'black', align = 'edge')

    return fig



def plot_results(event):
    global f4ab_prime, f4xb_prime, alpha, f4ab_std, f4xb_std, alpha_std, alpha_ratio_hist

    if all([el is not None for el in [f4ab_prime, f4xb_prime, alpha, f4ab_std, f4xb_std, alpha_std, alpha_ratio_hist]]):
        f4prime_fit_fig = plot_fit(f4ab_prime, f4xb_prime, alpha, f'{hybrid} = alpha {parent1} + (1 - alpha) {parent2}', f"f4'({parent1}, {parent2}; i, j)", f"f4'({hybrid}, {parent2}; i, j)")
        f4_fit_fig = plot_fit(f4ab_std, f4xb_std, alpha_std, f'{hybrid} = alpha {parent1} + (1 - alpha) {parent2}', f"f4({parent1}, {parent2}; i, j)", f"f4({hybrid}, {parent2}; i, j)")
        f4_ratio_histogram_fig = plot_histogram(alpha_ratio_hist, f'{hybrid} = alpha {parent1} + (1 - alpha) {parent2}', 'f4 ratio', 'Counts')

        f4prime_fit_pane.object = f4prime_fit_fig
        f4_fit_pane.object = f4_fit_fig
        f4_ratio_histogram_pane.object = f4_ratio_histogram_fig



plot_width_input.param.watch(plot_results, 'value')
plot_height_input.param.watch(plot_results, 'value')
plot_title_size_input.param.watch(plot_results, 'value')
plot_label_size_input.param.watch(plot_results, 'value')



def compute_results(event):
    global alpha_pre_jl, cosine_pre_jl, angle_pre_jl, percentage_pre_jl, f3_test
    global f4ab_prime, f4xb_prime, alpha, alpha_error
    global f4ab_std, f4xb_std, alpha_std, alpha_std_error
    global cosine_post_jl, angle_post_jl, percentage_post_jl
    global alpha_ratio, alpha_ratio_avg, alpha_ratio_std_dev, alpha_ratio_hist, num_cases

    hybrid_freqs = allele_frequencies[hybrid]
    parent1_freqs = allele_frequencies[parent1]
    parent2_freqs = allele_frequencies[parent2]
    aux_freqs = [allele_frequencies[pop] for pop in auxiliaries]

    alpha_pre_jl = mixing_coefficient_pre_jl(hybrid_freqs, parent1_freqs, parent2_freqs)

    cosine_pre_jl, angle_pre_jl, percentage_pre_jl = admixture_angle_pre_jl(hybrid_freqs, parent1_freqs, parent2_freqs)

    f3_test = f3(hybrid_freqs, parent1_freqs, parent2_freqs)

    f4ab_prime, f4xb_prime = f4_prime(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs)
    alpha, alpha_error = least_squares(f4ab_prime, f4xb_prime)

    f4ab_std, f4xb_std = f4_std(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs)
    alpha_std, alpha_std_error = least_squares(f4ab_std, f4xb_std)

    cosine_post_jl, angle_post_jl, percentage_post_jl = admixture_angle_post_jl(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs)

    alpha_ratio, alpha_ratio_avg, alpha_ratio_std_dev, alpha_ratio_hist, num_cases = f4_ratio(hybrid_freqs, parent1_freqs, parent2_freqs, aux_freqs)

    # Output

    num_aux_pops = len(auxiliaries)
    num_aux_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    text = '### Admixture model\n'
    text += f'`{hybrid} = {parent1} + {parent2}`\n'
    text += f'SNPs employed: {num_valid_alleles} / {num_alleles}\n'
    text += f'Auxiliary populations: {num_aux_pops}\n'
    text += f'Auxiliary pairs: {num_aux_pairs}\n'
    text += '### Admixture angle\n'
    text += '| Calculation | Cosine | Angle (deg) | Percentage of 180 deg |\n'
    text += '| --- | --- | --- | --- |\n'
    text += f'| Pre-JL | {cosine_pre_jl:7.4f} | {angle_pre_jl:7.2f} | {percentage_pre_jl:.1%} |\n'
    text += f'| Post-JL | {cosine_post_jl:7.4f} | {angle_post_jl:7.2f} | {percentage_post_jl:.1%} |\n'
    text += '### Mixing coefficient\n'
    text += '| Calculation | Alpha | Error (95% CI) |\n'
    text += '| --- | --- | --- |\n'
    text += f'| Pre-JL | {alpha_pre_jl:6.4f} | - |\n'
    text += f'| Post-JL (f4-prime, renormalized) | {alpha:6.4f} | {alpha_error:6.4f} |\n'
    text += f'| Post-JL NR (f4, standard) | {alpha_std:6.4f} | {alpha_std_error:6.4f} |\n'
    text += '### f4-ratio and f3 test\n'
    text += f'f4-ratio average if [0, 1]: {alpha_ratio_avg:6.4f} +/- {alpha_ratio_std_dev:6.4f} (95% CI), {num_cases} cases\n'
    text += f'Standard admixture test: f3(c1, c2; x) < 0 ? {f3_test:8.6f}'

    resulting_data_output.object = text

    plot_results(None)

    results_download_button.disabled = False
    f4_points_download_button.disabled = False



pn.bind(compute_results, compute_results_button, watch = True)



def save_population_allele_frequencies():
    file = io.StringIO()

    pops_width = max([len(name) for name in sel_pops])
    prec = 6
    col_width = max(prec + 7, pops_width)

    headers_format = ' '.join([f'{{{i}:^{col_width}}}' for i, pop in enumerate(sel_pops)])
    headers = headers_format.format(*sel_pops)
    file.write(headers + '\n')

    row_format = ' '.join([f'{{{i}: {col_width}.{prec}E}}' for i, pop in enumerate(sel_pops)])

    for allele_index in range(num_valid_alleles):
        row = [freqs[allele_index] for pop, freqs in allele_frequencies.items()]
        file.write(row_format.format(*row) + '\n')

    file.seek(0)

    now = datetime.now()
    name = now.strftime("frequencies_%Y-%m-%d_%Hh%Mm%Ss")
    freqs_download_button.filename = name + '.dat'

    return file



freqs_download_button.callback = save_population_allele_frequencies



def save_f4_points():
    file = io.StringIO()

    aux_pops_width = max([len(name) for name in auxiliaries])
    prec = 6
    col_width = prec + 7

    headers = '{0:^{col_width}} {1:^{col_width}} {2:^{col_width}} {3:^{col_width}} {4:^{col_width}} {5:^{aux_pops_width}} {6:^{aux_pops_width}}'.format('f4primeAB', 'f4primeXB', 'f4AB', 'f4XB', 'f4-ratio', 'Aux1', 'Aux2', col_width = col_width, aux_pops_width = aux_pops_width)
    file.write(headers + '\n')

    num_aux_pops = len(sel_pops[3:])

    index = 0

    for i in range(num_aux_pops):
        for j in range(i + 1, num_aux_pops):
            row = '{0: {col_width}.{prec}E} {1: {col_width}.{prec}E} {2: {col_width}.{prec}E} {3: {col_width}.{prec}E} {4: {col_width}.{prec}E} {5:{aux_pops_width}} {6:{aux_pops_width}}'.format(f4ab_prime[index], f4xb_prime[index], f4ab_std[index], f4xb_std[index], alpha_ratio[index], auxiliaries[i], auxiliaries[j], prec = prec, col_width = col_width, aux_pops_width = aux_pops_width)
            file.write(row + '\n')
            index += 1

    file.seek(0)

    now = datetime.now()
    name = now.strftime("f4_points_%Y-%m-%d_%Hh%Mm%Ss")
    f4_points_download_button.filename = name + '.dat'

    return file



f4_points_download_button.callback = save_f4_points



def save_admixture_data():
    num_aux_pops = len(auxiliaries)
    num_aux_pairs = int(num_aux_pops * (num_aux_pops - 1) / 2)

    file = io.StringIO()

    file.write(f'Admixture model: {hybrid} = {parent1} + {parent2}\n')
    file.write(f'SNPs employed: {num_valid_alleles} / {num_alleles}\n')
    file.write(f'Auxiliary populations: {num_aux_pops}\n')
    file.write(f'Auxiliary pairs: {num_aux_pairs}\n')
    file.write(f'Cos pre-JL:  {cosine_pre_jl:7.4f} ---> Angle pre-JL:  {angle_pre_jl:7.2f} deg vs 180 deg: {percentage_pre_jl:.1%}\n')
    file.write(f'Cos post-JL: {cosine_post_jl:7.4f} ---> Angle post-JL: {angle_post_jl:7.2f} deg vs 180 deg: {percentage_post_jl:.1%}\n')
    file.write(f'Alpha pre-JL:     {alpha_pre_jl:6.4f}\n')
    file.write(f'Alpha post-JL:    {alpha:6.4f} +/- {alpha_error:6.4f} (95% CI) (f4-prime, renormalized)\n')
    file.write(f'Alpha NR post-JL: {alpha_std:6.4f} +/- {alpha_std_error:6.4f} (95% CI) (f4, standard)\n')
    file.write(f'f4-ratio average if [0, 1]: {alpha_ratio_avg:6.4f} +/- {alpha_ratio_std_dev:6.4f} (95% CI), {num_cases} cases\n')
    file.write(f'Standard admixture test: f3(c1, c2; x) < 0 ? {f3_test:8.6f}')

    file.seek(0)

    now = datetime.now()
    name = now.strftime("results_%Y-%m-%d_%Hh%Mm%Ss")
    results_download_button.filename = name + '.dat'

    return file



results_download_button.callback = save_admixture_data



def admixture_model_card_width():
    return hybrid_select.width + 2 * hybrid_select.margin[1] + parent1_select.width + 2 * parent1_select.margin[1] + parent2_select.width + 2 * parent2_select.margin[1] + aux_pops_select.width + 2 * aux_pops_select.margin[1]

def admixture_model_card_height():
    print(hybrid_select.height)
    return hybrid_select.height + 2 * hybrid_select.margin[0]



# Cards and their layouts

card_margin = 10

alert_card = pn.Card(alert_pane, title = 'Messages', collapsible = True, margin = card_margin)

zip_file_dropper_card = pn.Card(zip_file_dropper, title = 'Upload ZIP with input triad', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})
dat_file_dropper_card = pn.Card(dat_file_dropper, title = 'Upload file with selected populations', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})
input_droppers_layout = pn.FlexBox(zip_file_dropper_card, dat_file_dropper_card, flex_direction = 'row')

avail_pops_card = pn.Card(avail_pops_filter, avail_pops_table, title = 'Available populations', margin = card_margin, styles = {'width': 'fit-content'})
sel_pops_card = pn.Card(reset_sel_pops_button, sel_pops_table, title = 'Selected populations', margin = card_margin, styles = {'width': 'fit-content'})
comp_card = pn.Card(compute_freqs_button, freqs_download_button, title = 'Actions', margin = card_margin, styles = {'width': 'fit-content'})

admixture_model_layout = pn.FlexBox(hybrid_select, parent1_select, parent2_select, aux_pops_select, flex_direction = 'row', justify_content = 'space-evenly')
admixture_model_card = pn.Card(admixture_model_layout, title = 'Admixture model', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})

actions_card = pn.Card(compute_results_button, f4_points_download_button, results_download_button, title = 'Actions', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})
plot_options_card = pn.Card(plot_width_input, plot_height_input, plot_title_size_input, plot_label_size_input, title = 'Plot options', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})
resulting_data_card = pn.Card(resulting_data_output, title = 'Results', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})

plots_layout = pn.FlexBox(f4prime_fit_pane, f4_fit_pane, f4_ratio_histogram_pane, flex_direction = 'row', justify_content = 'space-evenly')
plots_card = pn.Card(plots_layout, title = 'Plots', collapsible = False, margin = card_margin, styles = {'width': 'fit-content'})

# Layouts


input_files_layout = pn.Column(input_droppers_layout, load_files_button, name = 'Inputs', sizing_mode = 'stretch_width', styles = {'width': 'fit-content'})
select_pops_layout = pn.FlexBox(avail_pops_card, sel_pops_card, comp_card, name = 'Select populations', flex_direction = 'row', styles = {'width': 'fit-content'})
results_layout = pn.FlexBox(admixture_model_card, actions_card, plot_options_card, resulting_data_card, plots_card, name = 'Results', flex_direction = 'row', justify_content = 'space-evenly', styles = {'width': 'fit-content'})

# Tabs

tabs = pn.Tabs(input_files_layout, select_pops_layout, results_layout)

# Main layout

main_layout = pn.Column(alert_card, tabs, scroll = True)

# Template

template = pn.template.VanillaTemplate(title = 'Mixtum: The geometry of admixture in population genetics')
template.main.append(main_layout)
template.servable()

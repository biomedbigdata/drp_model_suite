import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import warnings

set_of_unmatched_cell_lines = set()
matched_cell_lines = dict()
renamed_cell_lines = dict()
no_match = set()


def create_cl_dict(df):
    # iterate over cellosaurus and make a dictionary mapping the 'ID' column to the 'AC' column and every cell line name
    # in the 'SY' column to the 'AC' column
    print('Creating cellosaurus dictionary ...')
    cellosaurus_ac_dict = {}
    cellosaurus_sy_dict = {}
    species_dict = {}
    for index, row in df.iterrows():
        # add species to species_dict
        species_dict[row['AC']] = row['OX']
        # check whether cellosaurus_dict[row['ID']] already exists, if not add it, if yes print the cell line name
        if row['ID'] not in cellosaurus_ac_dict and row['ID'] != '':
            cellosaurus_ac_dict[row['ID']] = row['AC']
        for cell_line_name in row['SY'].split('; '):
            # check whether cellosaurus_dict[cell_line_name] already exists, if not add it, if yes print the cell
            # line name
            if cell_line_name not in cellosaurus_sy_dict and cell_line_name != '':
                cellosaurus_sy_dict[cell_line_name] = row['AC']
    return cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict

def create_cl_dict_cell_passp(df):
    # iterate over cellosaurus and make a dictionary mapping the 'ID' column to the 'SID' IDs from the DR column if
    # it contains Cell_Model_Passport
    print('Creating cellosaurus dictionary ...')
    cellosaurus_sid_dict = {}
    species_dict = {}
    for index, row in df.iterrows():
        # add species to species_dict
        species_dict[row['AC']] = row['OX']
        # check whether the DR column contains Cell_Model_Passport
        if 'Cell_Model_Passport' in row['DR']:
            # split DR column by ',', iterate until you encounter an ID starting with SIDM
            for element in row['DR'].split(','):
                if element.startswith('SIDM'):
                    cellosaurus_sid_dict[element] = row['AC']
        else:
            continue
    return cellosaurus_sid_dict, species_dict


def map_to_cellosaurus(df, cl_dict_ac, cl_dict_sy, species_dict, output_path):
    # iterate over the cell line names in the dataframe. Try to get the cell line name from the cellosaurus
    # dictionary. If it exists, put it in a column called 'cellosaurus_id'. If it doesn't exist, print the cell line
    # name
    print('Mapping cell line names to cellosaurus IDs ...')
    for index, row in df.iterrows():
        try:
            df.loc[index, 'cellosaurus_id'] = cl_dict_ac[index]
            matched_cell_lines[index] = cl_dict_ac[index]
            species = species_dict.get(cl_dict_ac[index])
            # if 'Human' is not part of species string, warn
            if 'Human' not in species:
                warnings.warn(f'Cell line {cl_dict_ac[index]} matched to {index} is not human, but {species}.')
        except KeyError:
            try:
                df.loc[index, 'cellosaurus_id'] = cl_dict_sy[index]
                matched_cell_lines[index] = cl_dict_sy[index]
                species = species_dict.get(cl_dict_sy[index])
                # if 'Human' is not part of species string, warn
                if 'Human' not in species:
                    warnings.warn(f'Cell line {cl_dict_sy[index]} matched to {index} is not human, but {species}.')
                if index not in set_of_unmatched_cell_lines:
                    set_of_unmatched_cell_lines.add(index)
                    if SequenceMatcher(a=index.lower(), b=list(cl_dict_ac.keys())[list(cl_dict_ac.values()).index(cl_dict_sy[index])].lower()).ratio() < 0.7:
                        print(
                        f'no main match for {index}, matched it to {cl_dict_sy[index]} = {list(cl_dict_ac.keys())[list(cl_dict_ac.values()).index(cl_dict_sy[index])]}, '
                        f'but the similarity was rather low:'
                        f'{SequenceMatcher(a=index, b=list(cl_dict_ac.keys())[list(cl_dict_ac.values()).index(cl_dict_sy[index])]).ratio()}')
            except KeyError:
                print(f'no match at all for {index}')
                df.loc[index, 'cellosaurus_id'] = pd.NA
                no_match.add(index)

    # drop all rows where no cellosaurus ID could be found
    df = df.dropna(subset=['cellosaurus_id'])
    # save the gene expression dataframe with the cellosaurus IDs
    print('Saving dataframe with cellosaurus IDs ...')
    # make index to column 'cell_line_name' and make 'cellosaurus_id' the index
    df = df.reset_index().rename(columns={'index': 'cell_line_name'}).set_index('cellosaurus_id')
    df.to_csv(output_path)

def map_to_cellosaurus_model_passp(df, cl_dict_sid, species_dict, output_path, ignore_columns = []):
    # iterate over the cell line names in the dataframe. Try to get the cell line name from the cellosaurus
    # dictionary. If it exists, put it in a column called 'cellosaurus_id'. If it doesn't exist, print the cell line
    # name
    print('Mapping cell line names to cellosaurus IDs ...')
    # get all column names except the ones in ignore_columns
    columns = [col for col in df.columns if col not in ignore_columns]
    new_columns = list(df.columns.values)
    for cl_name in columns:
        try:
            new_columns[new_columns.index(cl_name)] = cl_dict_sid[cl_name]
            matched_cell_lines[cl_name] = cl_dict_sid[cl_name]
            species = species_dict.get(cl_dict_sid[cl_name])
            # if 'Human' is not part of species string, warn
            if 'Human' not in species:
                warnings.warn(f'Cell line {cl_dict_sid[cl_name]} matched to {cl_name} is not human, but {species}.')
        except KeyError:
            print(f'no match at all for {cl_name}')
            no_match.add(cl_name)

    # rename columns
    df.columns = new_columns
    # save the gene expression dataframe with the cellosaurus IDs
    print('Saving dataframe with cellosaurus IDs ...')
    # make first column index
    df = df.set_index(new_columns[0])
    df.to_csv(output_path)


def preprocess_gex():
    # read in gene expression dataframe
    print('Preprocessing gene expression dataframe ...')
    gex = pd.read_csv('cell_line_input/GDSC/gene_expression.csv')
    gex = gex.rename(columns={'Unnamed: 0': 'cell_line_name'})
    # make to index
    gex = gex.set_index('cell_line_name')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'K2': 'K2 [Human melanoma]',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'C32': 'C32 [Human melanoma]',
               '786-0': '786-O',
               'PC-3 [JPC-3]': 'PC-3',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'G-292 Clone A141B1': 'G-292 clone A141B1',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'SAT': 'SAT [Human HNSCC]',
               'HH': 'HH [Human lymphoma]',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'TK': 'TK [Human B-cell lymphoma]',
               'NOS-1': 'NOS-1 [Human osteosarcoma]'}
    gex = gex.rename(
        index=renamed)
    renamed_cell_lines.update(renamed)
    return gex


def preprocess_methylation():
    # read in methylation dataframe
    print('Preprocessing methylation dataframe ...')
    methylation = pd.read_csv('cell_line_input/GDSC/methylation.csv')
    methylation = methylation.rename(columns={'Unnamed: 0': 'cell_line_name'})
    # make to index
    methylation = methylation.set_index('cell_line_name')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'MO': 'Mo',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'CA-SKI': 'Ca Ski',
               'C32': 'C32 [Human melanoma]',
               'EOL-1-CELL': 'EoL-1',
               'GA-10-CLONE-4': 'GA-10 clone 4',
               'HH': 'HH [Human lymphoma]',
               'JIYOYEP-2003': 'Jiyoye',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'LC-2-AD': 'LC-2/ad',
               'LNCAP-CLONE-FGC': 'LNCaP clone FGC',
               'NO-11': 'Onda 11',
               'NO-10': 'Onda 10',
               'NTERA-S-CL-D1': 'NT2-D1',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'HUO-3N1': 'HuO-3N1',
               'CAR-1': 'CaR-1',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'OMC-1': 'OMC-1 [Human cervical carcinoma]',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'HEP3B2-1-7': 'Hep 3B2.1-7',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'G-292-CLONE-A141B1': 'G-292 clone A141B1',
               'HEP_G2': 'Hep-G2',
               'LC-1-SQ': 'LC-1/sq',
               'RERF-LC-SQ1': 'RERF-LC-Sq1',
               'SAT': 'SAT [Human HNSCC]',
               'TK': 'TK [Human B-cell lymphoma]',
               'PC-3_JPC-3': 'PC-3'}
    methylation = methylation.rename(
        index=renamed)
    renamed_cell_lines.update(renamed)
    return methylation


def preprocess_mutation():
    # read in mutation dataframe
    print('Preprocessing mutation dataframe ...')
    mutation = pd.read_csv('cell_line_input/GDSC/mutations.csv')
    mutation = mutation.rename(columns={'model_name': 'cell_line_name'})
    # make to index
    mutation = mutation.set_index('cell_line_name')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'K2': 'K2 [Human melanoma]',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'C32': 'C32 [Human melanoma]',
               'G-292-Clone-A141B1': 'G-292 clone A141B1',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'HH': 'HH [Human lymphoma]',
               'Hep3B2-1-7': 'Hep 3B2.1-7',
               'Hs-633T': 'Hs 633.T',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'OMC-1': 'OMC-1 [Human cervical carcinoma]',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'SAT': 'SAT [Human HNSCC]',
               'TALL-1': 'TALL-1 [Human adult T-ALL]',
               'TK': 'TK [Human B-cell lymphoma]'
               }
    mutation = mutation.rename(
        index=renamed)
    renamed_cell_lines.update(renamed)
    return mutation


def preprocess_cnv():
    # read in copy number variation dataframe
    print('Preprocessing copy number variation dataframe ...')
    cnv = pd.read_csv('cell_line_input/GDSC/copy_number_variation_gistic.csv')
    cnv = cnv.rename(columns={'model_name': 'cell_line_name'})
    # make to index
    cnv = cnv.set_index('cell_line_name')
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'K2': 'K2 [Human melanoma]',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'C32': 'C32 [Human melanoma]',
               'OMC-1': 'OMC-1 [Human cervical carcinoma]',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'TK': 'TK [Human B-cell lymphoma]',
               'SAT': 'SAT [Human HNSCC]',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'TALL-1': 'TALL-1 [Human adult T-ALL]',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'HH': 'HH [Human lymphoma]',
               'Hep3B2-1-7': 'Hep 3B2.1-7',
               'G-292-Clone-A141B1': 'G-292 clone A141B1'}
    cnv = cnv.rename(
        index=renamed)
    return cnv


def preprocess_binarized_drp():
    # read in drug response dataframe
    print('Preprocessing drug response dataframe ...')
    drp = pd.read_csv('response_output/GDSC/binarized_gdsc.csv')
    drp = drp.drop(0)
    drp = drp.rename(columns={'Screened Compounds:': 'cell_line_name'})
    # make to index
    drp = drp.set_index('cell_line_name')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'K2': 'K2 [Human melanoma]',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'C32': 'C32 [Human melanoma]',
               'SAT': 'SAT [Human HNSCC]',
               'TK': 'TK [Human B-cell lymphoma]',
               'HH': 'HH [Human lymphoma]',
               'G-292 Clone A141B1': 'G-292 clone A141B1',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'PC-3 [JPC-3]': 'PC-3',
               'OMC-1': 'OMC-1 [Human cervical carcinoma]',
               'TALL-1': 'TALL-1 [Human adult T-ALL]'
               }
    drp = drp.rename(
        index=renamed)
    renamed_cell_lines.update(renamed)
    return drp


def collapse_ln_ic50s(values):
    # for each value: take exponential of value. Take mean of all values. Take log of mean.
    return np.log(np.mean(np.exp(values)))


def preprocess_gdsc_1():
    # read in drug response dataframe
    print('Preprocessing drug response dataframe ...')
    drp = pd.read_csv('response_output/GDSC/response_GDSC1.csv')
    # drop all columns except 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50'
    drp = drp[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']]
    # collapse duplicate IC50 values
    len_before = len(drp)
    drp = drp.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).agg(collapse_ln_ic50s)
    print(f'Collapsed {len_before - len(drp)}/{len_before} duplicated ln(IC50) values = {len(drp)} unique values now.')
    drp = drp.reset_index()
    drp = drp.rename(columns={'CELL_LINE_NAME': 'cell_line_name'})
    # get long format into wide format: rows should be cell line names (CELL_LINE_NAME column), columns should be drug names (DRUG_NAME column), values are in LN_IC50 column
    drp = drp.pivot(index='cell_line_name', columns='DRUG_NAME', values='LN_IC50')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'JM1': 'JM-1',
               'HT55': 'HT-55',
               'K2': 'K2 [Human melanoma]',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'C32': 'C32 [Human melanoma]',
               'G-292-Clone-A141B1': 'G-292 clone A141B1',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'HH': 'HH [Human lymphoma]',
               'Hep3B2-1-7': 'Hep 3B2.1-7',
               'Hs-633T': 'Hs 633.T',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'ML-1': 'ML-1 [Human thyroid carcinoma]',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'NTERA-2-cl-D1': 'NT2-D1',
               'OMC-1': 'OMC-1 [Human cervical carcinoma]',
               'PC-3_[JPC-3]': 'PC-3',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'SAT': 'SAT [Human HNSCC]',
               'TALL-1': 'TALL-1 [Human adult T-ALL]',
               'TK': 'TK [Human B-cell lymphoma]'}
    drp = drp.rename(
        index=renamed)
    return drp


def preprocess_gdsc_2():
    # read in drug response dataframe
    print('Preprocessing drug response dataframe ...')
    drp = pd.read_csv('response_output/GDSC/response_GDSC2.csv')
    # drop all columns except 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50'
    drp = drp[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']]
    # collapse duplicate IC50 values
    len_before = len(drp)
    drp = drp.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).agg(collapse_ln_ic50s)
    print(f'Collapsed {len_before - len(drp)}/{len_before} duplicated ln(IC50) values = {len(drp)} unique values now.')
    drp = drp.reset_index()
    drp = drp.rename(columns={'CELL_LINE_NAME': 'cell_line_name'})
    # get long format into wide format: rows should be cell line names (CELL_LINE_NAME column), columns should be drug names (DRUG_NAME column), values are in LN_IC50 column
    drp = drp.pivot(index='cell_line_name', columns='DRUG_NAME', values='LN_IC50')
    # replace the cell line names, e.g., 'RCM-1' with 'RCM-1 [Human rectal adenocarcinoma]', 'C32' with 'C32 [Human melanoma]'
    renamed = {'HT55': 'HT-55',
               'MS-1': 'MS-1 [Human lung carcinoma]',
               'C32': 'C32 [Human melanoma]',
               'G-292-Clone-A141B1': 'G-292 clone A141B1',
               'HARA': 'HARA [Human squamous cell lung carcinoma]',
               'HH': 'HH [Human lymphoma]',
               'Hep3B2-1-7': 'Hep 3B2.1-7',
               'Hs-633T': 'Hs 633.T',
               'KS-1': 'KS-1 [Human glioblastoma]',
               'NOS-1': 'NOS-1 [Human osteosarcoma]',
               'NTERA-2-cl-D1': 'NT2-D1',
               'PC-3_[JPC-3]': 'PC-3',
               'RCM-1': 'RCM-1 [Human rectal adenocarcinoma]',
               'SAT': 'SAT [Human HNSCC]',
               'TK': 'TK [Human B-cell lymphoma]'}
    drp = drp.rename(
        index=renamed)
    renamed_cell_lines.update(renamed)
    return drp


def preprocess_sanger_ccle(tpm=True):
    # read in gene expression dataframe
    print('Preprocessing gene expression dataframe ...')
    if tpm:
        gex = pd.read_csv('cell_line_input/SangerCellModelPassports/sanger_tpm_ccle.csv', sep=',')
    else:
        gex = pd.read_csv('cell_line_input/SangerCellModelPassports/sanger_counts_ccle.csv', sep=',')
    return gex


def preprocess_sanger_sanger(tpm=True):
    # read in gene expression dataframe
    print('Preprocessing gene expression dataframe ...')
    if tpm:
        gex = pd.read_csv('cell_line_input/SangerCellModelPassports/sanger_tpm_sanger.csv', sep=',')
    else:
        gex = pd.read_csv('cell_line_input/SangerCellModelPassports/sanger_counts_sanger.csv', sep=',')
    return gex


if __name__ == '__main__':
    cellosaurus = pd.read_csv('mapping/cellosaurus_01_2024.csv')
    # replace all NaN values with empty strings
    cellosaurus = cellosaurus.fillna('')
    # create cellosaurus dictionary
    cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict = create_cl_dict(cellosaurus)
    # map gene expression cell line names to cellosaurus IDs
    gex = preprocess_gex()
    map_to_cellosaurus(gex, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'cell_line_input/GDSC/gene_expression_cellosaurus.csv')
    # map methylation cell line names to cellosaurus IDs
    met = preprocess_methylation()
    map_to_cellosaurus(met, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'cell_line_input/GDSC/methylation_cellosaurus.csv')
    # map mutation cell line names to cellosaurus IDs
    mut = preprocess_mutation()
    map_to_cellosaurus(mut, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict, 'cell_line_input/GDSC/mutations_cellosaurus.csv')
    # map copy number variation cell line names to cellosaurus IDs
    cnv = preprocess_cnv()
    map_to_cellosaurus(cnv, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'cell_line_input/GDSC/copy_number_variation_gistic_cellosaurus.csv')
    # map binarized drug response cell line names to cellosaurus IDs
    drp_bin = preprocess_binarized_drp()
    map_to_cellosaurus(drp_bin, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'response_output/GDSC/binarized_gdsc_cellosaurus.csv')
    # map drug response cell line names to cellosaurus IDs
    drp_gdsc_1 = preprocess_gdsc_1()
    map_to_cellosaurus(drp_gdsc_1, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'response_output/GDSC/response_GDSC1_cellosaurus.csv')
    # map drug response cell line names to cellosaurus IDs
    drp_gdsc_2 = preprocess_gdsc_2()
    map_to_cellosaurus(drp_gdsc_2, cellosaurus_ac_dict, cellosaurus_sy_dict, species_dict,
                       'response_output/GDSC/response_GDSC2_cellosaurus.csv')

    # export matched cell lines to csv file
    matched_cell_lines_df = pd.DataFrame.from_dict(matched_cell_lines, orient='index', columns=['cellosaurus_id'])
    matched_cell_lines_df.to_csv('mapping/matched_cell_lines.csv')

    # export unmatched cell lines to csv file
    unmatched_cell_lines_df = pd.DataFrame(list(no_match), columns=['cell_line_name'])
    unmatched_cell_lines_df.to_csv('mapping/unmatched_cell_lines.csv')

    # export renamed cell lines to csv file
    renamed_cell_lines_df = pd.DataFrame.from_dict(renamed_cell_lines, orient='index', columns=['cell_line_name'])
    renamed_cell_lines_df.to_csv('mapping/renamed_cell_lines.csv')

    cellosaurus_sid_dict, species_dict = create_cl_dict_cell_passp(cellosaurus)
    gex = preprocess_sanger_sanger(tpm=True)
    map_to_cellosaurus_model_passp(gex, cellosaurus_sid_dict, species_dict,
                                   'cell_line_input/SangerCellModelPassports/sanger_tpm_sanger_cvcl.csv',
                                   ignore_columns=['ensembl_gene_id', 'gene_symbol'])
    gex = preprocess_sanger_ccle(tpm=True)
    map_to_cellosaurus_model_passp(gex, cellosaurus_sid_dict, species_dict,
                                   'cell_line_input/SangerCellModelPassports/sanger_tpm_ccle_cvcl.csv',
                                   ignore_columns=['ensembl_gene_id', 'gene_symbol'])

    gex = preprocess_sanger_sanger(tpm=False)
    map_to_cellosaurus_model_passp(gex, cellosaurus_sid_dict, species_dict,
                                   'cell_line_input/SangerCellModelPassports/sanger_counts_sanger_cvcl.csv',
                                   ignore_columns=['ensembl_gene_id', 'gene_symbol'])
    gex = preprocess_sanger_ccle(tpm=False)
    map_to_cellosaurus_model_passp(gex, cellosaurus_sid_dict, species_dict,
                                   'cell_line_input/SangerCellModelPassports/sanger_counts_ccle_cvcl.csv',
                                   ignore_columns=['ensembl_gene_id', 'gene_symbol'])


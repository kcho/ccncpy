from __future__ import division
import pandas as pd
import nibabel as nb
import numpy as np
import re
from numpy import count_nonzero, mean
from multiprocessing import Pool

import os
from os.path import join, basename, isdir, isfile

import seaborn as sns
import matplotlib.pyplot as plt


def total_connectivity_estimation(seed_map):
    return seed_map.sum()

def relative_connectivity_estimation(total_connectivity, sum_connectivity):
    return total_connectivity/sum_connectivity

def sum_connectivity_estimation(total_connectivity_array):
    return total_connectivity_array.sum()

def maksed_mean_map(img_map, mask_map):
    return img_map[np.nonzero(mask_map)].mean()

def get_volume(img_map):
    return np.count_nonzero(img_map)

def get_map(f):
    return nb.load(f).get_data()

class get_subject_info:
    def __init__(self, param):
#         try:
        dataLoc, subject = param

        nuclei_dict = {"LPFC":1, "LTC":2, "MPFC":3, "MTC":4,
                       "OCC":5, "OFC":6, "PC":7, "SMC":8}


        subjDir = join(dataLoc, subject)
        dkiDir = join(dataLoc, subject, 'DKI')
        mk_map_fs = get_map(join(dkiDir, 'kmean_freesurfer_space.nii.gz'))
        mk_map_dki = get_map(join(dkiDir, 'kmean.nii'))

        # file information
        get_thr = lambda x: int(re.search('\d{1,2}', x).group(0)) if re.search('\d{1,2}', x) else 0
        get_space = lambda x: 'dki' if re.search('dki', x) else 'fs'
        get_side = lambda x: 'left' if re.search('lh|left', x) else 'right'
        get_matching_mk_map = lambda x: mk_map_dki if x=='dki' else mk_map_fs
        get_cortex = lambda x: re.search('lpfc|ltc|mpfc|mtc|occ|ofc|pc|smc', x, re.IGNORECASE).group(0)

        img_files = []
        for root, dirs, files in os.walk(subjDir):
            for f in files:
                if f.endswith('nii.gz'):
                    img_files.append(join(root, f))

        roi_files = [x for x in img_files if 'ROI' in x and re.search('lpfc|ltc|mpfc|mtc|occ|ofc|pc|smc', x, re.IGNORECASE)]
        seed_files = [x for x in img_files if re.search('segmentation_fnirt.+seeds_to', x)]
        #biggest_files = [x for x in img_files if re.search('biggest', x)]

        roi_df = pd.DataFrame()
        for roi_file in roi_files:
            roi_basename = basename(roi_file)

            thr = get_thr(roi_basename)
            space = get_space(roi_basename)
            side = get_side(roi_basename)
            cortex = get_cortex(roi_basename)
            roi_map = get_map(roi_file)
            mk_map = get_matching_mk_map(space)

            roi_volume = count_nonzero(roi_map)
            roi_mk = mean(mk_map[(mk_map!=0) & (roi_map>0)])

            df = pd.DataFrame({'subject':[subject],
                               'space':space,
                               'cortex':cortex,
                               'threshold':thr,
                               'side':side,
                               'cortex_volume':roi_volume,
                               'cortex_mk':roi_mk})
            roi_df = pd.concat([roi_df, df])

        seed_df = pd.DataFrame()
        for seed_file in seed_files:
            seed_basename = basename(seed_file)

            thr = get_thr(seed_basename)
            space = get_space(seed_basename)
            side = get_side(seed_basename)
            cortex = get_cortex(seed_basename)
            seed_map = get_map(seed_file)
            mk_map = get_matching_mk_map(space)

            seed_volume = count_nonzero(seed_map)
            seed_mk = mean(mk_map[(mk_map!=0) & (seed_map>0)])
            connectivity = sum(seed_map[seed_map!=0])

            df = pd.DataFrame({'subject':[subject],
                               'space':space,
                               'cortex':cortex,
                               'threshold':thr,
                               'side':side,
                               'seed_volume':seed_volume,
                               'connectivity':connectivity,
                               'seed_mk':seed_mk})

            #df = df.pivot_table(index=['subject','seed_name','space', 'cortex', 'side'],
                                #columns=['seed_volume', 'connectivity', 'seed_mk'],
                                #values='threshold',
                                #aggfunc=np.sum)

            seed_df = pd.concat([seed_df, df])


        self.seed_df_orig = seed_df
        seed_df = seed_df.pivot_table(index=['subject', 'space','side','cortex'], 
                                      columns='threshold', 
                                      values=['seed_mk', 'seed_volume', 'connectivity'], 
                                      aggfunc=np.sum).reset_index()
        subjectDf = pd.merge(roi_df, seed_df,
                             on=['subject', 'cortex', 'side', 'space'],
                             how='outer')

        subjectDf = subjectDf.sort_values(by=['subject','cortex', 'side', 'space'])

        self.roi_df = roi_df
        self.seed_df = seed_df
        self.subjectDf = subjectDf

if __name__=='__main__':

    dataLoc = '/Volume/CCNC_BI_3T/kcho/allData'

    # pool = Pool(processes=24)
    pool = Pool()
    subjects = [x for x in os.listdir(dataLoc) if x.startswith('NOR') or x.startswith('FEP') or x.startswith('CHR')]

    input_params = [(dataLoc, x) for x in subjects]
    outs = pool.map(get_subject_info, input_params)

    df = pd.concat([x.subjectDf for x in outs])
    get_columns_cleared = lambda col: '{}_{}'.format(col[0], col[1]) if type(col)==tuple else col
    df.columns = [get_columns_cleared(x) for x in df.columns]

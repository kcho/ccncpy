from __future__ import division
import pandas as pd
import nibabel as nb
import numpy as np
import re
from numpy import count_nonzero, mean
from multiprocessing import Pool

import os
from os.path import join, basename, isdir, isfile
import pickle

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
    '''
    True or False : thalamus, roi, biggest, seed
    f = get_subject_info(dataLoc, subject, thalamus, roi, biggest, seed)
    '''
    def get_biggest_information(self):
        biggest_df = pd.DataFrame()
        for biggest_file in self.biggest_files:
            biggest_basename = basename(biggest_file)

            thr = get_thr(biggest_basename)
            space = get_space(biggest_basename)
            side = get_side(biggest_file)
            biggest_map = get_map(biggest_file)
            mk_map = self.get_matching_mk_map(space)
            #mk_map = mk_map_fs

            for cortex, number in self.nuclei_dict.iteritems():
                biggest_volume = count_nonzero(biggest_map[biggest_map == number])
                biggest_mk = mean(mk_map[(mk_map != 0) & (biggest_map == number)])

                df = pd.DataFrame({'subject':[self.subject],
                                   'space':space,
                                   'cortex':cortex,
                                   'threshold':thr,
                                   'side':side,
                                   'biggest_volume':biggest_volume,
                                   'biggest_mk':biggest_mk})

                biggest_df = pd.concat([biggest_df, df])
        self.biggest_df = biggest_df

    def get_seed_information(self):
        seed_df = pd.DataFrame()
        for seed_file in self.seed_files:
            seed_basename = basename(seed_file)

            thr = get_thr(seed_basename)
            space = get_space(seed_basename)
            side = get_side(seed_basename)
            cortex = get_cortex(seed_basename)
            seed_map = get_map(seed_file)
            mk_map = self.get_matching_mk_map(space)

            seed_volume = count_nonzero(seed_map)
            seed_mk = mean(mk_map[(mk_map!=0) & (seed_map>0)])
            connectivity = sum(seed_map[seed_map!=0])

            df = pd.DataFrame({'subject':[self.subject],
                               'space':space,
                               'cortex':cortex,
                               'threshold':thr,
                               'side':side,
                               'seed_volume':seed_volume,
                               'connectivity':connectivity,
                               'seed_mk':seed_mk})


            seed_df = pd.concat([seed_df, df])
        self.seed_df = seed_df

    def get_roi_information(self):
        roi_df = pd.DataFrame()
        for roi_file in self.roi_files:
            roi_basename = basename(roi_file)

            thr = get_thr(roi_basename)
            space = get_space(roi_basename)
            side = get_side(roi_basename)
            cortex = get_cortex(roi_basename)
            roi_map = get_map(roi_file)
            mk_map = self.get_matching_mk_map(space)

            roi_volume = count_nonzero(roi_map)
            roi_mk = mean(mk_map[(mk_map!=0) & (roi_map>0)])

            df = pd.DataFrame({'subject':[self.subject],
                               'space':space,
                               'cortex':cortex,
                               #'threshold':thr,
                               'side':side,
                               'cortex_volume':roi_volume,
                               'cortex_mk':roi_mk})
            roi_df = pd.concat([roi_df, df])
        self.roi_df = roi_df

    def get_img_files(self):
        img_files = []
        for root, dirs, files in os.walk(self.subjDir):
            for f in files:
                if f.endswith('nii.gz'):
                    img_files.append(join(root, f))

        self.roi_files = [x for x in img_files if 'ROI' in x and re.search('lpfc|ltc|mpfc|mtc|occ|ofc|pc|smc', x, re.IGNORECASE)]
        self.thalamus_roi_files = [x for x in img_files if 'ROI' in x and re.search('thalamus.nii.gz', x, re.IGNORECASE)]
        self.seed_files = [x for x in img_files if re.search('segmentation_fnirt.+seeds_to', x)]
        self.biggest_files = [x for x in img_files if re.search('biggest', x)]

    def get_thalamus_information(self):
        thalamus_roi_df = pd.DataFrame()
        for roi_file in self.thalamus_roi_files:
            roi_basename = basename(roi_file)

            space = get_space(roi_basename)
            side = get_side(roi_basename)

            roi_map = get_map(roi_file)
            mk_map = self.get_matching_mk_map(space)

            roi_volume = count_nonzero(roi_map)
            roi_mk = mean(mk_map[(mk_map!=0) & (roi_map>0)])

            df = pd.DataFrame({'subject':[self.subject],
                               'space':space,
                               'side':side,
                               'thalamus_volume':roi_volume,
                               'thalamus_mk':roi_mk})
            thalamus_roi_df = pd.concat([thalamus_roi_df, df])
        self.thalamus_roi_df = thalamus_roi_df


    def __init__(self, param):
#         try:
        #dataLoc, subject  = param

        self.dataLoc, self.subject, self.thalamus, self.roi, self.biggest, self.seed = param
        print(self.subject)

        self.nuclei_dict = {"LPFC":1, "LTC":2, "MPFC":3, "MTC":4,
                            "OCC":5, "OFC":6, "PC":7, "SMC":8}

        self.subjDir = join(self.dataLoc, self.subject)
        self.dkiDir = join(self.subjDir, 'DKI')
        self.mk_map_fs = get_map(join(self.dkiDir, 'kmean_freesurfer_space.nii.gz'))
        self.mk_map_dki = get_map(join(self.dkiDir, 'kmean.nii'))

        # file information
        get_thr = lambda x: int(re.search('\d{1,2}', x).group(0)) if re.search('\d{1,2}', x) else 0
        get_space = lambda x: 'dki' if re.search('dki', x) else 'fs'
        get_side = lambda x: 'left' if re.search('lh|left', x) else 'right'
        self.get_matching_mk_map = lambda x: self.mk_map_dki if x=='dki' else self.mk_map_fs
        get_cortex = lambda x: re.search('lpfc|ltc|mpfc|mtc|occ|ofc|pc|smc', x, re.IGNORECASE).group(0)

        pickle_loc = join(self.dataLoc, self.subject, 'data.pkl')
        if isfile(pickle_loc):
            print(pickle_loc)
            with open(pickle_loc, 'rb') as f:
                self.subjectDf = pickle.load(f)
        else:
            self.get_img_files()

            thalamus_roi_pickle = join(self.dataLoc, self.subject, 'thalamus_roi.pkl')
            if self.thalamus and not isfile(thalamus_roi_pickle):
                self.get_thalamus_information()
                with open(thalamus_roi_pickle, 'wb') as f:
                    pickle.dump(self.thalamus_roi_df, f)
            elif not self.thalamus and not isfile(thalamus_roi_pickle):
                self.get_thalamus_information()
                with open(thalamus_roi_pickle, 'wb') as f:
                    pickle.dump(self.thalamus_roi_df, f)
            else:
                with open(thalamus_roi_pickle, 'rb') as f:
                    self.thalamus_roi_df = pickle.load(f)

            roi_pickle = join(self.dataLoc, self.subject, 'roi.pkl')
            if self.roi and not isfile(roi_pickle):
                self.get_roi_information()
                with open(roi_pickle, 'wb') as f:
                    pickle.dump(self.roi_df, f)
            elif not self.roi and not isfile(roi_pickle):
                self.get_roi_information()
                with open(roi_pickle, 'wb') as f:
                    pickle.dump(self.roi_df, f)
            else:
                with open(roi_pickle, 'rb') as f:
                    self.roi_df = pickle.load(f)

            biggest_pickle = join(self.dataLoc, self.subject, 'biggest.pkl')
            if self.biggest and not isfile(biggest_pickle):
                self.get_biggest_information()
                with open(biggest_pickle, 'wb') as f:
                    pickle.dump(self.biggest_df, f)
            elif not self.biggest and not isfile(biggest_pickle):
                self.get_biggest_information()
                with open(biggest_pickle, 'wb') as f:
                    pickle.dump(self.biggest_df, f)
            else:
                with open(biggest_pickle, 'rb') as f:
                    self.biggest_df = pickle.load(f)

            seed_pickle = join(self.dataLoc, self.subject, 'seed.pkl')
            if self.seed and not isfile(seed_pickle):
                self.get_seed_information()
                with open(seed_pickle, 'wb') as f:
                    pickle.dump(self.seed_df, f)
            elif not self.seed and not isfile(seed_pickle):
                self.get_seed_information()
                with open(seed_pickle, 'wb') as f:
                    pickle.dump(self.seed_df, f)
            else:
                with open(seed_pickle, 'rb') as f:
                    self.seed_df = pickle.load(f)

            seed_df = pd.merge(self.seed_df, self.biggest_df,
                                 on=['subject', 'threshold', 'cortex', 'side', 'space'],
                                 how='left')

            seed_df = seed_df.pivot_table(index=['subject', 'space','side','cortex'], 
                                          columns='threshold', 
                                          values=['seed_mk', 'seed_volume', 'connectivity', 'biggest_volume', 'biggest_mk'], 
                                          aggfunc=np.sum).reset_index()

            subjectDf = pd.merge(self.roi_df, seed_df,
                                 on=['subject', 'cortex', 'side', 'space'],
                                 how='outer')

            subjectDf = pd.merge(self.thalamus_roi_df, subjectDf,
                                 on=['subject', 'side', 'space'],
                                 how='right')

            subjectDf = subjectDf.sort_values(by=['subject', 'cortex', 'side', 'space'])

            self.subjectDf = subjectDf

            with open(pickle_loc, 'wb') as f:
                pickle.dump(subjectDf, f)


    def __str__(self):
        return str(self.subject)

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

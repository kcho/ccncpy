from os.path import join
import numpy as np
import nibabel as nb
import os
from xml.dom import minidom
from collections import Counter
import itertools
import pandas as pd
import re
from matplotlib.colors import from_levels_and_colors


class mniT1Data:
    def __init__(self):
        self.mni = join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_1mm_brain.nii.gz')
        self.mni_img = nb.load(self.mni)
        self.mni_data = self.mni_img.get_data()


        self.mni_mask = join(os.environ['FSLDIR'], 'data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
        self.mni_mask_img = nb.load(self.mni_mask)
        self.mni_mask_data = self.mni_mask_img.get_data()
        
class havardOxfordThalamusData:
    def __init__(self):
        self.mni_subcortex = join(os.environ['FSLDIR'], 
                                  'data/atlases',
                                  'HarvardOxford',
                                  'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz')
        self.mni_subcortex_img = nb.load(self.mni_subcortex)
        self.mni_subcortex_data = self.mni_subcortex_img.get_data()
        
        self.prob_img = nb.load(join(os.environ['FSLDIR'],
                                     'data/atlases',
                                     'HarvardOxford',
                                     'HarvardOxford-cort-prob-1mm.nii.gz'))
        self.prob_data = self.prob_img.get_data()

        self.mni_subcortex_data_masked = np.ma.masked_where(~np.isin(self.mni_subcortex_data, [15, 4]), self.mni_subcortex_data)


class talairachData:
    def __init__(self):
        self.talairach = join(os.environ['FSLDIR'], 
                              'data/atlases/Talairach/Talairach-labels-1mm.nii.gz')
        self.talairach_img = nb.load(self.talairach)
        self.talairach_data = self.talairach_img.get_data()



class talairachLabels:
    def __init__(self):
        self.xmldoc = minidom.parse(join(os.environ['FSLDIR'],
                                         'data/atlases/Talairach.xml'))
        self.itemlist = self.xmldoc.getElementsByTagName('label')

        self.talairach_dict = {}
        for s in self.itemlist:
            label_number = int(s.attributes['index'].value)
            region_name = s.childNodes[0].data
            self.talairach_dict[label_number] = region_name

        self.talairach_df = pd.DataFrame.from_dict(self.talairach_dict, orient='index')
        self.talairach_df.index.name='label number'


class talairachThalamusLabels(talairachLabels):
    def __init__(self):
        super().__init__()
        # select thalamus labels
        self.talairach_thalamus_df = self.talairach_df[self.talairach_df[0].str.contains('Sub-lobar.Thalamus')]

        # side and name extracted to columns
        self.talairach_thalamus_df['side'] = self.talairach_thalamus_df[0].str.split(' ').str[0]
        self.talairach_thalamus_df['name'] = self.talairach_thalamus_df[0].str.split('.').str[-1]
        self.remove_nucleus_from_name = lambda x: re.sub(' Nucleus', '', x) if x.endswith('Nucleus') else x
        self.talairach_thalamus_df['name'] = self.talairach_thalamus_df.name.map(self.remove_nucleus_from_name)

        # re-number nuclei
        self.nuclei_list = self.talairach_thalamus_df['name'].unique()
        self.name_num_dict = dict(zip(self.nuclei_list, range(1, len(self.nuclei_list)+1)))
        self.talairach_thalamus_df['roi_number'] = self.talairach_thalamus_df['name'].map(self.name_num_dict)

        

    
class talirachThalamusLabelsColor(talairach_thalamus_labels):
    def __init__(self):
        super().__init__()        
        self.colors = np.array([[0.80526917, 0.76246004, 0.71236253],
                                [0.73105507, 0.82655098, 0.89263817],
                                [0.69931578, 0.6526525 , 0.5823189 ],
                                [0.74857169, 0.81418701, 0.65601533],
                                [0.86296798, 0.90864072, 0.64738471],
                                [0.68078472, 0.57431632, 0.63717782],
                                [0.93066726, 0.93946834, 0.59296891],
                                [0.79797146, 0.55516047, 0.78058996],
                                [0.82512093, 0.58636063, 0.63810387],
                                [0.73687016, 0.86102455, 0.59222474],
                                [0.56751623, 0.81363333, 0.75638909],
                                [0.92659012, 0.56316328, 0.67795806]])
        self.levels = np.arange(12 + 1) - 0.5
        self.cmap, self.norm = from_levels_and_colors(self.levels, self.colors)
        self.cmap.set_bad('white', 0)

        # talairach_thalamus_df.loc[:, 'colors'] = list(colors)
        self.color_df = pd.DataFrame({'number':np.arange(1,len(self.colors)+1),
                                      'colors':list(self.colors)})

        self.talairach_thalamus_df = pd.merge(self.talairach_thalamus_df.reset_index(),
                                              self.color_df,
                                              left_on = 'roi_number',
                                              right_on = 'number', how='left')

        self.number_to_colors_dict = self.color_df.set_index('number').to_dict()['colors']
        self.talairach_label_to_color_dict = self.talairach_thalamus_df.set_index('name').to_dict()['colors']



class talairachThalamusImage(talirachThalamusLabelsColor, talairachData):
    def __init__(self):
        talairachData.__init__(self)
        super().__init__()        
        self.orig_num_to_new_dict = self.talairach_thalamus_df['roi_number'].to_dict()
#         print(self.talairach_data)
        self.talairach_thalamus_data = self.talairach_data.copy()
        # zero other areas apart from the thalamus
        self.talairach_thalamus_data[~np.isin(self.talairach_data, 
                                              self.talairach_thalamus_df['label number'].values)] = 0

        # change the values of the labels
        for origNum, newNum in self.talairach_thalamus_df.set_index('label number').roi_number.to_dict().items():
            np.place(self.talairach_thalamus_data, self.talairach_thalamus_data==origNum, newNum)


class behrensMask():
    def __init__(self):
        self.behrens_mask_loc = join(os.environ['FSLDIR'], 
                                'data/atlases/Thalamus', 
                                'Thalamus-maxprob-thr25-1mm.nii.gz')
        self.behrens_mask_img = nb.load(self.behrens_mask_loc)
        self.behrens_mask_data = self.behrens_mask_img.get_data()
        self.behrens_mask_data_masked = np.ma.masked_where(self.behrens_mask_data == 0, self.behrens_mask_data)

class behrensMaskColor():
    def __init__(self):
        self.behrensRoiNumDict = {"Primary motor":1,
                                  "Sensory":2,
                                  "Occipital":3,
                                  "Pre-frontal":4,
                                  "Pre-motor":5,
                                  "Posterior parietal":6,
                                  "Temporal":7}
        self.behrensNumRoiDict = {value : key for key,value in self.behrensRoiNumDict.items()}

        self.behrens_colors = np.array([[0.87192607, 0.66041753, 0.91142589],
                                        [0.80917289, 0.88120141, 0.88958628],
                                        [0.6416239 , 0.86516164, 0.71139534],
                                        [0.56606377, 0.61487396, 0.73786088],
                                        [0.86076639, 0.63772511, 0.67666386],
                                        [0.66172476, 0.85218288, 0.62232168],
                                        [0.75858755, 0.83385981, 0.81687487]])
        self.behrens_levels = np.arange(7 + 1) - 0.5
        self.behrens_cmap, self.behrens_norm = from_levels_and_colors(self.behrens_levels, self.behrens_colors)
        self.behrens_cmap.set_bad('white', 0)

        self.behrens_color_df = pd.DataFrame({'number':np.arange(1, 8),
                                              'colors':list(self.behrens_colors)})
        self.behrens_color_df['name']  = self.behrens_color_df['number'].map(self.behrensNumRoiDict)
        self.behrens_label_to_color_dict = self.behrens_color_df.set_index('name').to_dict()['colors']

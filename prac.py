from __future__ import unicode_literals
import numpy as np
import pandas as pd
import nibabel as nb
import os
from os.path import join, basename, dirname, isfile, isdir
import re
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.plotting as plotting

# pipeline
import nipype
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.fsl import UnaryMaths
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces import fsl
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms

from multiprocessing import Pool
import GPUtil # CCNC GPU server
import sys
sys.path.append('/Volumes/CCNC_4T/psyscan/thalamus_project')
import roiExtraction

# Ipython notebook
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backends.backend_pdf import PdfPages

import math

from ipywidgets import *
#get_ipython().run_line_magic('matplotlib', 'notebook') # Jupyter notebook


def plot_4d_dwi_pdf(img_loc, outname, z_gap=3, ncols=15, page_num=7):
    '''
    Plot 4d dwi data. Save as PDF.
    - vmin / vmax percentile
    '''
    print('Summary pdf of {}'.format(img_loc))
    print('PDF saved at {}'.format(outname))

    img_data = nb.load(img_loc).get_data()

    # Match brightness of the diffusion weighted volumes
    vmin = img_data[:,:,:,-1].min() # vmin and vmax estimation in the last volume
    vmax = img_data[:,:,:,-1].max() # which is likely be non-b0 image

    # Initialise fig and ax
    # Columns : different z-slices
    # Rows : differnt volumes of the dwi data

    vol_num_s = 0
    pdf = PdfPages(outname)
    nrows = math.ceil(img_data.shape[3] / page_num)
    for page in range(0, page_num): # for each page
        fig, axes = plt.subplots(ncols=ncols, 
                                 nrows=nrows,
                                 figsize=(11.69, 8.27), 
                                 dpi=300)

        fig.suptitle('{}'.format(img_loc), 
                     fontsize=14, 
                     fontweight='bold')

        for vol_num, row_axes in enumerate(axes, vol_num_s): # for each row
            slice_num = 5
            for col_num, ax in enumerate(row_axes): # for each column
                img = ax.imshow(img_data[:,:,slice_num,vol_num], vmin=vmin, vmax=vmax)#, aspect=img_data[0]/img_data[1])
                ax.set_axis_off()
                slice_num += z_gap
            row_axes[0].text(0, 0.5, 
                             vol_num+1,
                             verticalalignment='center', horizontalalignment='right',
                             rotation=90, 
                             transform=row_axes[0].transAxes,
                             fontsize=15)

        vol_num_s = vol_num+1
        plt.subplots_adjust(wspace=0, hspace=0)
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close()
    pdf.close()

def plot_3d_dwi_pdf(img_loc, outname, ncols=15):
    '''
    Plot 3d dwi data. Save as PDF.
    - vmin / vmax percentile
    '''
    print('Summary pdf of {}'.format(img_loc))
    print('PDF saved at {}'.format(outname))

    img_data = nb.load(img_loc).get_data()

    # Match brightness of the diffusion weighted volumes
    vmin = img_data[:,:,:].min() # vmin and vmax estimation in the last volume
    vmax = img_data[:,:,:].max() # which is likely be non-b0 image

    # Initialise fig and ax
    # Columns : different z-slices
    # Rows : differnt volumes of the dwi data

    pdf = PdfPages(outname)
    nrows = math.ceil(img_data.shape[2] / ncols)

    fig, axes = plt.subplots(ncols=ncols, 
                             nrows=nrows, 
                             figsize=(11.69, 8.27), 
                             dpi=300)

    fig.suptitle('{}'.format(img_loc), 
                 fontsize=14, 
                 fontweight='bold')

    slice_num = 5
    for slice_num, ax in enumerate(np.ravel(axes): # for each axes
        img = ax.imshow(img_data[:,:,slice_num], vmin=vmin, vmax=vmax)#, aspect=img_data[0]/img_data[1])
        ax.set_axis_off()
        slice_num += z_gap
        axes[slice_num].text(0.5, 0, 
                             vol_num+1,
                             verticalalignment='center', horizontalalignment='center',
                             #rotation=90, 
                             transform=row_axes[0].transAxes,
                             fontsize=10)
        slice_num+=1
    plt.subplots_adjust(wspace=0, hspace=0)
    pdf.savefig(fig)  # saves the current figure into a pdf page
    plt.close()
    pdf.close()

def plot_4d_dwi(img_data:np.array):
    '''
    Plot 4d data with the widgets
    - vmin / vmax percentile
    - slice number
    - volume number
    '''

    # Match brightness of the diffusion weighted volumes
    vmin = img_data[:,:,:,-1].min() # vmin and vmax estimation in the last volume
    vmax = img_data[:,:,:,-1].max() # which is likely be non-b0 image

    # initialise fig and ax
    fig, ax = plt.subplots(ncols=1, figsize=(5,5))
    img = ax.imshow(img_data[:,:,0,0], vmin=vmin, vmax=vmax)

    axcolor = 'lightgoldenrodyellow'
    ax_z_slice = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_t_slice = plt.axes([0.25, 0.00, 0.65, 0.03], facecolor=axcolor)

    z_slice = Slider(ax_z_slice, 'Z-slice', 0, img_data.shape[2]-1, valinit=0, valstep=1, valfmt='%1.0f')
    t_slice = Slider(ax_t_slice, 'T-slice', 0, img_data.shape[3]-1, valinit=0, valstep=1, valfmt='%1.0f')


    def update(val):
        z = int(z_slice.val)
        t = int(t_slice.val)
        img.set_data(img_data[:,:, z, t])
        fig.canvas.draw_idle()
    z_slice.on_changed(update)
    t_slice.on_changed(update)

    plt.show()

def plot_3d_nifti(img_data:np.array):
    '''
    Plot 3d data with the widgets
    - slice number
    '''

    # initialise fig and ax
    fig, ax = plt.subplots(ncols=1)
    img = ax.imshow(img_data[:,:,30])

    axcolor = 'lightgoldenrodyellow'
    ax_z_slice = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    z_slice = Slider(ax_z_slice, 'Z-slice', 0, img_data.shape[2]-1, valinit=0, valstep=1, valfmt='%1.0f')


    def update(val):
        z = int(z_slice.val)
        img.set_data(img_data[:,:, z])
        fig.canvas.draw_idle()
    z_slice.on_changed(update)

    plt.show()

def plot_bet_out(nodif_img_data:np.array, img_data:np.array):
    '''
    Plot 3d data with the widgets
    - slice number
    '''

    # initialise fig and ax
    fig, ax = plt.subplots(ncols=1)
    nodif_img = ax.imshow(nodif_img_data[:,:,30], cmap='hot')
    bet_img = np.ma.masked_where(img_data<=0, img_data)
    img = ax.imshow(bet_img[:,:,30])

    axcolor = 'lightgoldenrodyellow'
    ax_z_slice = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    z_slice = Slider(ax_z_slice, 'Z-slice', 0, img_data.shape[2]-1, valinit=0, valstep=1, valfmt='%1.0f')

    def update(val):
        z = int(z_slice.val)
        nodif_img.set_data(nodif_img_data[:,:,z])
        img.set_data(bet_img[:,:, z])
        fig.canvas.draw_idle()
    z_slice.on_changed(update)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, (str(self.bet_f-.05), self.bet_f, str(self.bet_f+0.05)), active=1)

    def colorfunc(label, bet_f):
        bet_f = float(label)
        self.run_bet()
        bet_data = nb.load(self.nodif_bet).get_data()
        img.set_data(bet_img[:,:, z])
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)
    plt.show()

def plot_motion(motion_array:np.array):
    '''
    motion_array : (n,2) matrix
    absolute and relative motion parameters from eddy
    '''
    fig, ax = plt.subplots(ncols=1)
    ax.plot(motion_array)
    fig.show()

def check_two_images(img1:str, img2:str):
    '''
    Plots imshow graph with both img1 and img2
    img1, img2 : nifti image location
    '''
    img1_data = nb.load(img1).get_data()
    img2_data = nb.load(img2).get_data()

    initial_slice_num = int(img1_data.shape[2] / 2)
    fig,ax = plt.subplots(ncols=1)

    img_0 = ax.imshow(img2_data[:,:,initial_slice_num])
    img_1 = ax.imshow(img1_data[:,:,initial_slice_num], cmap='hot', alpha=0.2)
    fig.show()

    def update(z=80):
        img_0.set_data(img2_data[:,:, z])
        img_1.set_data(img1_data[:,:, z])
        fig.canvas.draw()
    z_slice = widgets.IntSlider(min=0,
                                max=img1_data.shape[2]-1,
                                value=initial_slice_num,
                                description='Z slice')
    interact(update, z=z_slice);

def check_roi_overlays(args):
    '''
    Plots imshow graph with both img1 and img2
    img1, img2 : nifti image location
    '''
    img1_data = nb.load(args[0]).get_data()

    roi_data = np.zeros_like(img1_data)
    for num, img_loc in enumerate(args[1:],1):
        print(img_loc)
        single_roi_data = nb.load(img_loc).get_data()
        roi_data = np.add(roi_data, num*single_roi_data, casting="unsafe")

    roi_data = np.ma.masked_where(roi_data == 0, roi_data)

    initial_slice_num = int(img1_data.shape[2] / 2)
    fig,ax = plt.subplots(ncols=1)

    img_0 = ax.imshow(img1_data[:,:,initial_slice_num])
    img_1 = ax.imshow(roi_data[:,:,initial_slice_num], cmap='jet')
    fig.show()

    def update(z=80):
        img_0.set_data(img1_data[:,:, z])
        img_1.set_data(roi_data[:,:, z])
        fig.canvas.draw()
    z_slice = widgets.IntSlider(min=0,
                                max=img1_data.shape[2]-1,
                                value=initial_slice_num,
                                description='Z slice')
    interact(update, z=z_slice);

def check_roi_overlays_bilateral(args):
    '''
    Plots imshow graph with both img1 and img2
    img1, img2 : nifti image location
    '''
    img1_data = nb.load(args[0]).get_data()

    roi_data = np.zeros_like(img1_data)
    for num, img_loc in enumerate(args[1:],1):
        print(img_loc)
        single_roi_data = nb.load(img_loc).get_data()
        roi_data = np.add(roi_data, single_roi_data, casting="unsafe")

    roi_data = np.ma.masked_where(roi_data == 0, roi_data)

    initial_slice_num = int(img1_data.shape[2] / 2)
    fig,ax = plt.subplots(ncols=1)

    img_0 = ax.imshow(img1_data[:,:,initial_slice_num])
    img_1 = ax.imshow(roi_data[:,:,initial_slice_num], cmap='jet')
    fig.show()

    def update(z=80):
        img_0.set_data(img1_data[:,:, z])
        img_1.set_data(roi_data[:,:, z])
        fig.canvas.draw()
    z_slice = widgets.IntSlider(min=0,
                                max=img1_data.shape[2]-1,
                                value=initial_slice_num,
                                description='Z slice')
    interact(update, z=z_slice);

class mniSettings:
    def __init__(self):
        # ---
        # MNI
        # ---
        self.mni_fa_1mm = join(os.environ['FSLDIR'], 
                               'data/standard/FMRIB58_FA_1mm.nii.gz')
        self.mni_t1_1mm = join(os.environ['FSLDIR'], 
                               'data/standard/MNI152_T1_1mm.nii.gz')
        self.mni_t1_1mm_brain = join(os.environ['FSLDIR'], 
                                     'data/standard/MNI152_T1_1mm_brain.nii.gz')

        self.template_shortname_dict = {'Unclassified':'Outside',
                                        'Middle cerebellar peduncle':'MCP',
                                        'Pontine crossing tract (a part of MCP)':'P-MCP',
                                        'Genu of corpus callosum':'G-CC',
                                        'Body of corpus callosum':'B-CC',
                                        'Splenium of corpus callosum':'S-CC',
                                        'Fornix (column and body of fornix)':'FORNIX',
                                        'Corticospinal tract':'CST',
                                        'Medial lemniscus':'ML',
                                        'Inferior cerebellar peduncle':'ICP',
                                        'Superior cerebellar peduncle':'SCP',
                                        'Cerebral peduncle':'CP',
                                        'Anterior limb of internal capsule':'ALIC',
                                        'Posterior limb of internal capsule':'PLIC',
                                        'Retrolenticular part of internal capsule':'RL-ALIC',
                                        'Anterior corona radiata':'ACR',
                                        'Superior corona radiata':'SCR',
                                        'Posterior corona radiata':'PCR',
                                        'Posterior thalamic radiation (include optic radiation)':'PTR',
                                        'Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus)':'SS',
                                        'External capsule':'EC',
                                        'Cingulum (cingulate gyrus)':'CG',
                                        'Cingulum (hippocampus)':'h-CG',
                                        'Fornix (cres) / Stria terminalis (can not be resolved with current resolution)':'FORNIX/STR',
                                        'Superior longitudinal fasciculus':'SLF',
                                        'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule)':'SFOF',
                                        'Uncinate fasciculus':'UF',
                                        'Tapetum':'Tapetum',
                                        'Anterior thalamic radiation':'ATR',
                                        'Corticospinal tract':'CT',
                                        'Forceps major':'F-major',
                                        'Forceps minor':'F-minor',
                                        'Inferior fronto-occipital fasciculus':'IFOF',
                                        'Inferior longitudinal fasciculus':'ILF',
                                        'Superior longitudinal fasciculus (temporal part)':'SLF-t'}

class psyscanSettings(mniSettings):
    def __init__(self, subject_dir):
        super().__init__()
        # site information
        self.site = basename(subject_dir)[4:6]
        self.site_dict = {'01':'Melbourne', '02':'Vienna',
                          '03':'Copenhagen', '04':'Heidelberg (gottingen)',
                          '05':'Marburg', '06':'Galway',
                          '07':'Tel Hashomer', '08':'Naples',
                          '09':'Amsterdam', '10':'Maastricht',
                          '11':'Utrecht', '12':'Cantabria',
                          '13':'Madrid', '14':'Zurich',
                          '15':'Edinburgh', '16':'London',
                          '17':'Toronto', '18':'Seoul',
                          '19':'Sao Paulo', '20':'Hong King'}
        try:
            self.site_name = self.site_dict[self.site]
        except:
            self.site_name = 'other'
        echo_spacing_dict = {'01':0.56,
                             '02':0.69,
                             '04':0.53,
                             '05':0.53,
                             '08':0.53,
                             '10':0.5,
                             '15':0.53,
                             '16':0.596,
                             '17':0.596,
                             '18':0.53}
        self.group = basename(subject_dir)[3]
        try:
            self.echo_spacing = self.echo_spacing_dict[self.site]
        except:
            self.echo_spacing = 0.5

class fsSettings(psyscanSettings):
    def __init__(self, subject_dir):
        super().__init__(subject_dir)

        self.fs_dir = join(subject_dir, 'FREESURFER')
        self.reg_dir = join(subject_dir, 'registration')
        self.roi_dir = join(subject_dir, 'ROI')

        try:
            os.mkdir(self.reg_dir)
        except:
            pass

        try:
            os.mkdir(self.roi_dir)
        except:
            pass

        # ----------
        # Freesurfer 
        # ----------
        # recon-all has been done in a different server
        self.fs_mri_dir = join(self.fs_dir, 'mri')
        self.t1_mgz = join(self.fs_mri_dir, 'T1.mgz')
        self.t1 = join(self.fs_mri_dir, 'T1.nii.gz')
        self.t1_brain_mgz = join(self.fs_mri_dir, 'brain.mgz')
        self.t1_brain = join(self.fs_mri_dir, 'brain.nii.gz')
        

        # check data existence
        if not all([isfile(x) for x in [self.t1_mgz, self.t1_brain_mgz]]):
            print('{} : Problem in the initial data')

        if not all([isfile(x) for x in [self.t1, self.t1_brain]]):
            self.mgz_to_nii()

        self.t1_brain_recip = join(self.fs_mri_dir, 'brain_recip.nii.gz')

        # -------------
        # FS to MNI
        # FLIRT & FNIRT
        # -------------
        self.t1_to_mni_flirt_mat = join(self.reg_dir, 'fs_to_mni_flirt.mat')
        self.t1_to_mni_fnirt = join(self.reg_dir, 'fs_to_mni_warp_coeff.nii.gz')
        self.t1_to_mni_fnirt_img = join(self.reg_dir, 'fs_to_mni_warp.nii.gz')

    def mgz_to_nii(self):
        '''
        Convert brain.mgz to brain.nii
        in Freesurfer/mri directory
        '''
        mc = MRIConvert()
        mc.inputs.in_file = self.t1_brain_mgz
        mc.inputs.out_file = self.t1_brain
        mc.inputs.out_type = 'niigz'
        mc.run()
        
        mc = MRIConvert()
        mc.inputs.in_file = self.t1_mgz
        mc.inputs.out_file = self.t1
        mc.inputs.out_type = 'niigz'
        mc.run()

    def t1_to_mni_registration(self):
        '''
        FS/mri/brain.nii.gz --> MNI
        FLIRT, then FNIRT
        '''
        flt = fsl.FLIRT()#bins=256, cost_func='mutualinfo')
        flt.inputs.in_file = self.t1_brain
        flt.inputs.reference = self.mni_t1_1mm_brain
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.out_matrix_file = self.t1_to_mni_flirt_mat
        flt.inputs.out_file = re.sub('mat', 'nii.gz', self.t1_to_mni_flirt_mat)
        flt.inputs.dof = 12
        flt.inputs.interp = 'trilinear'
        flt.inputs.searchr_x = [-180, 180]
        flt.inputs.searchr_y = [-180, 180]
        flt.inputs.searchr_z = [-180, 180]
        print(flt.cmdline)
        flt.run()
    
        fnt = fsl.FNIRT()#bins=256, cost_func='mutualinfo')
        fnt.inputs.in_file = self.t1
        fnt.inputs.ref_file = self.mni_t1_1mm
        fnt.inputs.affine_file = self.t1_to_mni_flirt_mat
        fnt.inputs.fieldcoeff_file = self.t1_to_mni_fnirt
        fnt.inputs.warped_file = self.t1_to_mni_fnirt_img
        print(fnt.cmdline)
        fnt.run()


    def invert_t1(self):
        '''
        1/T1 intensities, for SyN application
        '''
        inverter = UnaryMaths()
        inverter.inputs.in_file = self.t1_brain
        inverter.inputs.out_file = self.t1_brain_recip
        inverter.inputs.operation = 'recip'
        inverter.run()       

class dtiSettings(psyscanSettings):
    def __init__(self, subject_dir:str):
        super().__init__(subject_dir)
        self.subject_dir = subject_dir
        self.dti_dir = join(subject_dir, 'DTI')

        # ---
        # DTI
        # ---
        self.dwi_data = join(self.dti_dir, 'DTI.nii.gz')
        self.bvals = join(self.dti_dir, 'bvals')
        self.bvecs_initial = join(self.dti_dir, 'bvecs')

        # check data existence
        if not all([isfile(x) for x in [self.dwi_data, self.bvals, self.bvecs_initial]]):
            print('{} : Problem in the initial data'.format(
                basename(self.subject_dir)))

        # Preproc
        self.nodif = join(self.dti_dir, 'nodif.nii.gz')

        self.bet_f = 0.2 # bet f value initialise
        self.nodif_bet = join(self.dti_dir, 'nodif_brain.nii.gz')
        self.nodif_bet_mask = join(self.dti_dir, 'nodif_brain_mask.nii.gz')

        self.FA_map_MNI_name = join(self.dti_dir, 'DTI_FA_MNI')
        self.FA_mni = join(self.dti_dir, 'DTI_FA_MNI.nii.gz')

        if self.check_shapes():
            if not isfile(self.nodif):
                self.extract_b0()
            if not all([isfile (x) for x in [self.nodif_bet, self.nodif_bet_mask]]):
                self.run_bet()

        # Eddy out
        self.dwi_data_eddy_out = join(self.dti_dir, 'eddy_out.nii.gz')
        self.bvecs_eddy_out = join(self.dti_dir, 'eddy_out.eddy_rotated_bvecs')
        self.motion_rms = join(self.dti_dir, 'eddy_out.eddy_restricted_movement_rms')

        # Postproc
        self.FA = join(self.dti_dir, 'DTI_FA.nii.gz')
        self.FA_unwarpped = join(self.dti_dir, 'DTI_unwarpped_FA.nii.gz')

        # --------
        # Bedpostx
        # --------
        self.bedpostx_prep_dir = join(subject_dir, 'DTI_preprocessed')
        self.bedpostx_dir = join(subject_dir, 'DTI_preprocessed.bedpostX')

    def check_shapes(self):
        dwi_img = nb.load(self.dwi_data)
        dwi_data = dwi_img.get_data()

        to_print = []
        to_print.append(basename(self.subject_dir))
        to_print.append('-----------')
        to_print.append('DWI shape : {}'.format(dwi_data.shape))
        
        bvecs = np.loadtxt(self.bvecs_initial)
        to_print.append('Bvector shape : {}'.format(bvecs.shape))

        bvals = np.loadtxt(self.bvals)
        to_print.append('Bvals shape : {}'.format(bvals.shape))
        to_print.append('-----------')

        #try:
            #if all([len(dwi_data.shape) == 4,
                    #dwi_data.shape[0] >= 128,
                    #dwi_data.shape[1] >= 128,
                    #dwi_data.shape[2] > 50,
                    #dwi_data.shape[3] > 60,
                    #bvecs.shape[0] == 3,
                    #bvecs.shape[1] == dwi_data.shape[3],
                    #bvals.shape[0] == dwi_data.shape[3]]) == True:
                #return True
            #else:
                #print([len(dwi_data.shape) == 4,
                    #dwi_data.shape[0] >= 128,
                    #dwi_data.shape[1] >= 128,
                    #dwi_data.shape[2] > 50,
                    #dwi_data.shape[3] > 60,
                    #bvecs.shape[0] == 3,
                    #bvecs.shape[1] == dwi_data.shape[3],
                    #bvals.shape[0] == dwi_data.shape[3]])
                #return to_print
        #except:
            #return to_print

    def extract_b0(self):
        fslroi = ExtractROI(in_file=self.dwi_data, 
                            roi_file=self.nodif, 
                            t_min=0, 
                            t_size=1)
        fslroi.run()

    def run_bet(self):
        btr = fsl.BET(in_file = self.nodif,
                      frac = self.bet_f,
                      out_file = self.nodif_bet,
                      mask = True)
        btr.run()

    def check_outputs(self):
        self.extracted_roi_check()

    def check_raw_dwi(self):
        dwi_img = nb.load(self.dwi_data)
        dwi_data = dwi_img.get_data()
        plot_4d_dwi(dwi_data)

    def check_bet(self):
        nodif_bet_img = nb.load(self.nodif_bet)
        nodif_bet_data = nodif_bet_img.get_data()
        nodif_data = nb.load(self.nodif).get_data()
        #plot_bet_out(, nodif_bet_data)
        #plot_3d_nifti(nodif_bet_data)
        # initialise fig and ax
        fig, ax = plt.subplots(ncols=1)
        nodif_img = ax.imshow(nodif_data[:,:,30], cmap='hot')
        self.bet_data = np.ma.masked_where(nodif_bet_data<=0, nodif_bet_data)
        img = ax.imshow(self.bet_data[:,:,30])

        axcolor = 'lightgoldenrodyellow'
        ax_z_slice = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        z_slice = Slider(ax_z_slice, 'Z-slice', 0, nodif_data.shape[2]-1, valinit=30, valstep=1, valfmt='%1.0f')

        def update(val):
            z = int(z_slice.val)
            nodif_img.set_data(nodif_data[:,:,z])
            img.set_data(self.bet_data[:,:, z])
            fig.canvas.draw_idle()
        z_slice.on_changed(update)


        rax = plt.axes([0.025, 0.10, 0.15, 0.8], facecolor=axcolor)
        f_vals = [x for x in np.arange(0.0, 0.5, 0.05)]
        f_vals = ['{:.2f}'.format(x) for x in f_vals]
        #"{:.10f}".format(f)

        radio = RadioButtons(rax, f_vals, active=7)

        def colorfunc(label):
            self.bet_f = float(label)
            print(self.bet_f)
            self.run_bet()
            bet_data = nb.load(self.nodif_bet).get_data()
            self.bet_data = np.ma.masked_where(bet_data <= 0, bet_data)
            img.set_data(self.bet_data[:,:, int(z_slice.val)])
            #rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
            #radio = RadioButtons(rax, (str(self.bet_f-.05), self.bet_f, str(self.bet_f+0.05)), active=1)
            fig.canvas.draw_idle()

        radio.on_clicked(colorfunc)
        fig.show()
        
        print("Final bet f value is : {}".format(self.bet_f))
        self.final_bet_f = self.bet_f
        with open(join(self.dti_dir, 'final_bet_f.txt'), 'w') as f:
            f.write(str(self.final_bet_f))

    def eddy_settings(self):
        # index
        data_img = nb.load(self.dwi_data)
        self.index_array = np.tile(1, data_img.shape[-1])
        self.index_loc = join(self.dti_dir, 'index.txt')

        # acqp
        self.acqp_num = (128-1) * self.echo_spacing * 0.001
        self.acqp_line = '0 -1 0 {}'.format(self.acqp_num)
        self.acqp_loc = join(self.dti_dir, 'acqp.txt')

        # eddy_command
        self.eddy_command = 'eddy_cuda8.0 \
                --imain={data} \
                --mask={mask} \
                --index={index} \
                --acqp={acqp} \
                --bvecs={bvecs} \
                --bvals={bvals} \
                --out={out} \
                --repol'.format(data=self.dwi_data,
                                mask=self.nodif_bet_mask,
                                index=self.index_loc,
                                acqp=self.acqp_loc,

                                bvecs=self.bvecs_initial,
                                bvals=self.bvals,
                                out=join(self.dti_dir,
                                         'eddy_out'))
        self.eddy_command = re.sub('\s+', ' ', self.eddy_command)

    def write_index(self):
        np.savetxt(self.index_loc, self.index_array,
                   fmt='%d', newline=' ')
        
    def write_acqp(self):
        with open(self.acqp_loc, 'w') as f:
            f.write(self.acqp_line)

    def run_eddy(self, gpu_num=0):
        self.eddy_settings()
        self.write_index()
        self.write_acqp()

        # get most efficient gpu number
        if GPUtil.getGPUs()[gpu_num].load == 0:
            pass
        else:
            deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.5, attempts=1, interval=1900, verbose=False)
            gpu_num = deviceID[0]

        self.gpu_command = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_num) + self.eddy_command
        print(self.gpu_command)
        print(os.popen(self.gpu_command).read())

    def check_motion(self):
        motion_array = np.loadtxt()
        plot_motion(motion_array)

    def bedpostx_gpu(self, gpu_num=0):
        new_dti_dir = join(dirname(self.dti_dir), 'DTI_preprocessed')
        # copy eddy processed files
        os.mkdir(new_dti_dir)

        for initial_name, new_name in zip(['bvals', 'eddy_out.eddy_rotated_bvecs', 'eddy_out.nii.gz', 'nodif_brain_mask.nii.gz'],
                                          ['bvals', 'bvecs', 'data.nii.gz', 'nodif_brain_mask.nii.gz']):
            shutil.copy(join(self.dti_dir, initial_name),
                        join(new_dti_dir, new_name))

        if GPUtil.getGPUs()[gpu_num].load == 0:
            pass
        else:
            deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.5, attempts=1, interval=1900, verbose=False)
            gpu_num = deviceID[0]

        command = 'CUDA_VISIBLE_DEVICES={gpu_num} bedpostx_gpu {dtiDir}'.format(
            gpu_num = gpu_num, dtiDir = new_dti_dir)
        print(command)
        os.popen(command).read()

    def check_FA_registration(self):
        plot_3d_nifti(nb.load(self.FA_mni).get_data())
        
    def extract_bundle_FAs(self):
        jhu_label = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz')
        jhu_label_img = nb.load(jhu_label)
        jhu_label_data = jhu_label_img.get_data()

        FA_mni_img = nb.load(self.FA_mni)
        FA_mni_data = FA_mni_img.get_data()

        mean_FA_dict = {}
        for label_num in np.unique(jhu_label_data):
            masked_FA = np.ma.masked_where(jhu_label_data!=label_num, FA_mni_data)
            mean_FA_dict[label_num] = masked_FA[masked_FA.nonzero()].mean()

        self.jhu_label_mean_FA_dict = mean_FA_dict
        
        # save as csv file
        mean_FA_dict.to_csv(join(self.dti_dir, 'JHU_label_mean_FA.txt'))
        
        jhu_tracts = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
        jhu_tracts_img = nb.load(jhu_tracts)
        jhu_tracts_data = jhu_tracts_img.get_data()

        mean_FA_tracts_dict = {}
        for tract_num in np.unique(jhu_tracts_data):
            masked_tract_FA = np.ma.masked_where(jhu_tracts_data!=tract_num, FA_mni_data)
            mean_FA_tracts_dict[tract_num] = masked_tract_FA[masked_tract_FA.nonzero()].mean()

        self.jhu_tracts_mean_FA_dict = mean_FA_tracts_dict
        
        # save as csv file

    def make_FA_data_frame(self):
        df_jhu_label = pd.DataFrame.from_dict(self.jhu_label_mean_FA_dict, orient='index').reset_index()
        df_jhu_label.columns = ['ROI_number', 'FA']
        df_jhu_label['Template'] = 'JHU_label'

        with open(join(os.environ['FSLDIR'], 'data/atlases/JHU-labels.xml'), 'r') as f:
            JHU_label_xml = f.read()

        jhu_label_num_label = dict(re.findall("index=\"(\d{1,2})\".+>(.+)</label>", JHU_label_xml))
        df_jhu_label['ROI'] = df_jhu_label['ROI_number'].astype('str').map(jhu_label_num_label)


        df_jhu_tract = pd.DataFrame.from_dict(self.jhu_tracts_mean_FA_dict, orient='index').reset_index()
        df_jhu_tract.columns = ['ROI_number', 'FA']
        df_jhu_tract['Template'] = 'JHU_tract'

        with open(join(os.environ['FSLDIR'], 'data/atlases/JHU-tracts.xml'), 'r') as f:
            JHU_tracts_xml = f.read()

        jhu_tract_num_label = dict(re.findall("index=\"(\d{1,2})\".+>(.+)</label>", JHU_tracts_xml))
        jhu_tract_num_label_zero_match = {}
        jhu_tract_num_label_zero_match[0] = 'Outside'
        for key, value in jhu_tract_num_label.items():
            jhu_tract_num_label_zero_match[int(key)+1] = value
        df_jhu_tract['ROI'] = df_jhu_tract['ROI_number'].map(jhu_tract_num_label_zero_match)

        JHU_FA_df = pd.concat([df_jhu_label, df_jhu_tract])
        
        def get_side(roi_name):
            try:
                if roi_name.endswith('R'):
                    return 'Right'
                elif roi_name.endswith('L'):
                    return 'Left'
                else:
                    return 'Middle'
            except:
                return 'Middle'

        JHU_FA_df['side'] = JHU_FA_df['ROI'].apply(get_side)
        JHU_FA_df.to_csv(join(self.dtidir, 'JHU_FA.txt'))
        
        def side_remove(roi_name):
            side_removed_roi_name = re.sub(' R| L', '', roi_name)
            return side_removed_roi_name
            
        JHU_FA_df['side_removed'] = JHU_FA_df['ROI'].apply(side_remove)
        JHU_FA_df['short_name'] = JHU_FA_df['side_removed'].map(template_shortname_dict)
        
        self.JHU_FA_df = JHU_FA_df
        
    def plot_JHU_FA(self):
        fig, axes = plt.subplots(ncols=3, nrows=2, 
                                 figsize=(10, 10))

        #        L M R
        #   JHU  o o o
        #   JHU  o o o

        for num, (group, table) in enumerate(self.JHU_FA_df.groupby(['Template', 'side'])):
            ax = np.ravel(axes)[num]
            sns.barplot(table.short_name,
                        table.FA,
                        ax=ax)#.set(axis_bgcolor='w')
            ax.set_title(group[1])
            ax.set_xlabel('')
            ax.set_ylabel('')
            #ax.set_xticks(rotation='vertical')
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
#                 tick.set_fontsize(5)
            

        np.ravel(axes)[0].set_ylabel('JHU labels FA')
        np.ravel(axes)[3].set_ylabel('JHU tracts FA')
        fig.tight_layout()
        fig.show()

    def check_label_overlay(self):
        jhu_label = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz')
        jhu_label_img = nb.load(jhu_label)
        jhu_label_data = jhu_label_img.get_data()

        jhu_tracts = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
        jhu_tracts_img = nb.load(jhu_tracts)
        jhu_tracts_data = jhu_tracts_img.get_data()
        
        FA_mni_img = nb.load(self.FA_mni)
        FA_mni_data = FA_mni_img.get_data()
    
        fig,axes = plt.subplots(ncols=2)
        img_0=axes[0].imshow(FA_mni_data[:,:,80])
        img_1=axes[0].imshow(np.ma.masked_where(jhu_label_data==0, jhu_label_data)[:,:,80])
        img_2=axes[1].imshow(FA_mni_data[:,:,80])
        img_3=axes[1].imshow(np.ma.masked_where(jhu_tracts_data==0, jhu_tracts_data)[:,:,80])
        fig.show()
        
        def update(z=80):
            img_0.set_data(FA_mni_data[:,:, z])
            img_1.set_data(np.ma.masked_where(jhu_label_data==0, jhu_label_data)[:,:, z])
            img_2.set_data(FA_mni_data[:,:, z])
            img_3.set_data(np.ma.masked_where(jhu_tracts_data==0, jhu_tracts_data)[:,:, z])
            fig.canvas.draw()

        z_slice = widgets.IntSlider(min=0, 
                                      max=FA_mni_data.shape[2]-1, 
                                      value=80, 
                                      description='Z slice')
        interact(update, z=z_slice);
 
    def plot_single_jhu(self, template, side, roi_name):
        gb = self.JHU_FA_df.groupby(['Template', 'side', 'short_name'])
        roi_series = gb.get_group(('JHU_'+template, side, roi_name)).iloc[0]
        roi_num = roi_series.ROI_number
        roi_full_name = roi_series.ROI
        
        jhu_tracts = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
        jhu_tracts_img = nb.load(jhu_tracts)
        jhu_tracts_data = jhu_tracts_img.get_data()
        
        FA_mni_img = nb.load(self.FA_mni)
        FA_mni_data = FA_mni_img.get_data()
    
        roi_img = nb.Nifti2Image((jhu_tracts_data==roi_num).astype(int),
                                 affine=jhu_tracts_img.affine)
        plotting.plot_roi(roi_img, bg_img=FA_mni_img,
                          title=roi_full_name)

        plt.show()

class antsSynSettings(dtiSettings, fsSettings):
    '''
    f = antsSynSettings(subject_dir)

    f.bedpostx_gpu_unwarpped()
    f.dtifit_unwarpped()
    '''
    def __init__(self, subject_dir):
        super().__init__(subject_dir)

        # --------
        # DTI
        # Ants SyN
        # --------
        self.dwi_data_unwarpped = join(self.dti_dir, 'DTI_unwarpped.nii.gz')
        self.nodif_bet_mask_unwarpped = join(self.dti_dir, 'nodif_brain_mask_unwarpped.nii.gz')

        # --------
        # Bedpostx
        # --------
        self.bedpostx_prep_dir_unwarp = join(subject_dir, 'DTI_preprocessed_unwarp')
        self.bedpostx_dir_unwarp = join(subject_dir, 'DTI_preprocessed_unwarp.bedpostX')

        # -----
        # flirt 
        # -----
        # Freesurfer/mri/brain.mgz : moving
        # DTI/nodif_brain.nii.gz : ref
        self.t1_to_dti_flirt_mat = join(self.reg_dir, 'fs_to_nodif.mat')
        self.t1_to_dti_flirt_out = join(self.reg_dir, 'fs_to_nodif.nii.gz')

        if not isfile(self.t1_to_dti_flirt_out):
            self.t1_to_dti_registration()

        self.t1_to_dti_flirt_mat_c3d = join(self.reg_dir, 'fs_to_nodif_c3d.txt')
                                        
        # ----------------
        # antsRegistration
        # ----------------
        # FS T1_recip : moving
        # DTI/nodif.nii.gz : ref
        self.t1_to_dti_ants_prefix = join(self.reg_dir, 'ants')
        self.t1_to_dti_ants_warp = self.t1_to_dti_ants_prefix + '1Warp.nii.gz'
        self.t1_to_dti_ants_mat = self.t1_to_dti_ants_prefix + '0GenericAffine.mat'

    def dti_to_mni_through_t1_warp(self):
        aw = fsl.ApplyWarp()
        aw.inputs.in_file = self.FA
        aw.inputs.ref_file = self.mni_t1_1mm_brain
        aw.inputs.field_file = self.t1_to_mni_fnirt
        print(aw.cmdline)
        res = aw.run() 
        
    def t1_to_dti_registration(self):
        flt = fsl.FLIRT()#bins=256, cost_func='mutualinfo')
        flt.inputs.in_file = self.t1_brain
        flt.inputs.reference = self.nodif_bet
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.out_matrix_file = self.t1_to_dti_flirt_mat
#         flt.inputs.out_file = False
        flt.inputs.out_file = self.t1_to_dti_flirt_out
        flt.inputs.dof = 12
        flt.inputs.interp = 'trilinear'
        flt.inputs.searchr_x = [-180, 180]
        flt.inputs.searchr_y = [-180, 180]
        flt.inputs.searchr_z = [-180, 180]
        print(flt.cmdline)
        flt.run()

    def t1_to_dti_registration_c3d(self):
        command = '/usr/local/c3d-1.1.0-Linux-x86_64/bin/c3d_affine_tool \
                        -src {src} \
                        -ref {ref} \
                        {mat} \
                        -fsl2ras -oitk {matout}'.format(src=self.t1_brain,
                                                        ref=self.nodif_bet,
                                                        mat=self.t1_to_dti_flirt_mat,
                                                        matout=self.t1_to_dti_flirt_mat_c3d)
#         print(re.sub('\s', '', command))
        print(command)
        os.popen(command).read()
        
    def ants_unwarping(self):
#         -r [{c3d_conv_b0_to_t1_mat},1] \
# antsRegistration -d 3 -r ../../data/PSYF05004/registration/b0_to_t1_vtk.txt -m CC[../../data/PSYF05004/FREESURFER/mri/brain_recip.nii.gz,../../data/PSYF05004/DTI/nodif.nii.gz,1,4] -c [100x70x20,1e-7,5] -t SyN[0.1,3,0] -f 4x2x1 -s 2x1x0vox -g 0.1x1.0x0.1 -u 1 -z 1 --winsorize-image-intensities [0.005, 0.995] -v -o ../../data/PSYF05004/registration/ants
#         reg = Registration()
#         reg.inputs.dimension = 3
        
#         reg.inputs.fixed_image = self.t1_brain_recip
#         reg.inputs.moving_image = self.nodif
#         reg.inputs.initial_moving_transform = self.t1_to_dti_flirt_mat_c3d
#         reg.inputs.invert_initial_moving_transform = True
#         reg.inputs.output_transform_prefix = self.t1_to_dti_ants_prefix
#         reg.inputs.metric = ['CC']
#         reg.inputs.metric_weight = [1]
#         reg.inputs.radius_or_number_of_bins = [4]
#         reg.inputs.number_of_iterations = [[100, 70, 20]]
#         reg.inputs.convergence_threshold = [1.e-7]
#         reg.inputs.convergence_window_size = [5]
#         reg.inputs.transforms = ['SyN']
#         reg.inputs.transform_parameters = [(0.1, 3, 0)]
#         reg.inputs.shrink_factors = [[4,2,1]]
#         reg.inputs.smoothing_sigmas = [[2,1,0]]
#         reg.inputs.sigma_units = ['vox']
#         reg.inputs.restrict_deformation = [[0, 1.0, 0]]
#         reg.inputs.collapse_output_transforms = True
#         reg.inputs.winsorize_lower_quantile = 0.005
#         reg.inputs.winsorize_upper_quantile = 0.995
#         reg.inputs.verbose = True
#         print(re.sub('\ --', ' \\\n\t--', reg.cmdline))
#         os.popen(reg.cmdline).read()
# #         reg.run()
        command = '/home/kangik/bin/ants/bin/antsRegistration -d 3 \
        -r [{t1_to_dti_mat_c3dconv}, 1] \
        -m CC[{t1_recip},{nodif},1,4] \
        -c [100x70x20,1e-7,5] \
        -t SyN[0.1,3,0] \
        -f 4x2x1 -s 2x1x0vox -g 0.1x1.0x0.1 -u 1 \
        -z 1 --winsorize-image-intensities [0.005, 0.995] \
        -o {outname}'.format(t1_to_dti_mat_c3dconv=self.t1_to_dti_flirt_mat_c3d,
                             t1_recip=self.t1_brain_recip,
                             nodif=self.nodif,
                             outname=self.t1_to_dti_ants_prefix)
        print(re.sub('\s+',' ', command))
        print(os.popen(command).read())
        
    def apply_ants_unwarping(self):
        at = ApplyTransforms()
        at.inputs.dimension = 3
        at.inputs.input_image_type = 3
        at.inputs.input_image = self.dwi_data_eddy_out
        at.inputs.reference_image = self.nodif
        at.inputs.output_image = self.dwi_data_unwarpped
        at.inputs.transforms = [self.t1_to_dti_ants_warp, self.t1_to_dti_ants_mat] #adding identity makes no difference
        print(re.sub('\ --', ' \\\n\t--', at.cmdline))
        os.popen('/home/kangik/bin/ants/bin/'+at.cmdline).read()
#         at.run()
        
        at = ApplyTransforms()
        at.inputs.dimension = 3
#         at.inputs.input_image_type = 2
        at.inputs.input_image = self.nodif_bet_mask
        at.inputs.interpolation = 'NearestNeighbor'
        at.inputs.reference_image = self.nodif
        at.inputs.output_image = self.nodif_bet_mask_unwarpped
        at.inputs.transforms = [self.t1_to_dti_ants_warp, self.t1_to_dti_ants_mat] #adding identity makes no difference
        print(re.sub('\ --', ' \\\n\t--', at.cmdline))
        os.popen('/home/kangik/bin/ants/bin/'+at.cmdline).read()
#         at.run()
    def ants_unwarping_check(self):
        check_two_images(self.nodif_bet_mask_unwarpped, self.nodif)
        
    def bedpostx_gpu_unwarpped(self, gpu_num=0):
        # copy eddy processed files
        try:
            os.mkdir(self.bedpostx_prep_dir_unwarp)
        except:
            shutil.rmtree(self.bedpostx_prep_dir_unwarp)
            shutil.rmtree(self.bedpostx_dir_unwarp)
            os.mkdir(self.bedpostx_prep_dir_unwarp)
#         self.nodif_bet_mask_unwarpped = join(self.dti_dir, 'nodif_brain_mask_unwarpped.nii.gz')

        for initial_name, new_name in zip(['bvals', 'eddy_out.eddy_rotated_bvecs', 'DTI_unwarpped.nii.gz', 'nodif_brain_mask_unwarpped.nii.gz'],
                                          ['bvals', 'bvecs', 'data.nii.gz', 'nodif_brain_mask.nii.gz']):
            shutil.copy(join(self.dti_dir, initial_name),
                        join(self.bedpostx_prep_dir_unwarp, new_name))

        if GPUtil.getGPUs()[gpu_num].load == 0:
            pass
        else:
            deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.5, attempts=1, interval=1900, verbose=False)
            gpu_num = deviceID[0]

        command = 'CUDA_VISIBLE_DEVICES={gpu_num} bedpostx_gpu {dtiDir}'.format(
            gpu_num = gpu_num, 
            dtiDir = self.bedpostx_prep_dir_unwarp)
        print(command)
        print(os.popen(command).read())

    def dtifit_unwarpped(self):
        dtifit_command = 'dtifit -k {data} -o {outname} -m {mask} -r {bvecs} -b {bvals}'.format(
                           data=self.dwi_data_unwarpped, #join(self.dti_dir, 'eddy_out.nii.gz'),
                           outname=join(self.dti_dir, 'DTI_unwarpped'),
                           mask=self.nodif_bet_mask_unwarpped,
                           bvecs=self.bvecs_eddy_out,
                           bvals=self.bvals)
        os.popen(dtifit_command).read()

class tcTractSettings(antsSynSettings):
    def __init__(self, subject_dir:str):
        super().__init__(subject_dir)
        # ---------------------------------
        # FS to DTI
        # Single warp file for tractography
        # ---------------------------------
        # flirt mat + ants registration coeff merged
        self.t1_to_dti_warp_and_mat = join(self.reg_dir, 'fs_to_dti_warp_and_mat.nii.gz')
        self.t1_to_dti_warp_and_mat_inv = join(self.reg_dir, 'fs_to_dti_warp_and_mat_inv.nii.gz')
        self.seg_dir = join(subject_dir, 'segmentation')
        self.cortices = ['LPFC', 'LTC', 'MPFC', 'MTC', 'OCC', 'OFC', 'PC', 'SMC']

    def extract_rois(self):
        '''
        Extract the thalamus and eight cortices
        '''
        roiExtraction.roiExtraction(self.subject_dir, 'ROI', self.fs_dir)
        
    def extracted_roi_check(self):
        rois = [x+'.nii.gz' for x in ['thalamus']+self.cortices]
        rois = ['lh_'+x for x in rois]+['rh_'+x for x in rois]
        rois = [self.t1_brain] + [join(self.roi_dir, x) for x in rois]
        check_roi_overlays(rois[:])

    def extracted_roi_check_specific(self, roi_name='rh_thalamus.nii.gz'):
        check_two_images(join(self.roi_dir, roi_name), self.t1_brain)

    def make_seed_space_to_DTI_warps(self):
        command = 'convertwarp \
                    --ref={nodif_brain} \
                    --warp1={t1_to_dti_warp} \
                    --premat={t1_to_dti_flirt} \
                    --out={out}'.format(nodif_brain=self.nodif,
                                        t1_to_dti_warp=self.t1_to_dti_ants_warp,
                                        t1_to_dti_flirt=self.t1_to_dti_flirt_mat,
                                        out=self.t1_to_dti_warp_and_mat)
        print(command)
        print(os.popen(command).read())
        
        command = 'invwarp \
                    -r {t1_brain} \
                    -o {out} \
                    -w {t1_to_dti_warp_and_mat}'.format(t1_brain=self.t1_brain,
                                                        out=self.t1_to_dti_warp_and_mat_inv,
                                                        t1_to_dti_warp_and_mat=self.t1_to_dti_warp_and_mat)
        print(command)
        print(os.popen(command).read())

    def thalamo_cortical_tractography(self, gpu_num=0):
        try:
            os.mkdir(join(self.subject_dir, 'segmentation'))
        except:
            pass
        
        for side, sside in zip(['left', 'right'], ['lh', 'rh']):
            segdir = join(self.subject_dir, 'segmentation', side)
            try:
                print(segdir)
                shutil.rmtree(segdir)
            except:
                pass
            try:
                os.mkdir(segdir)
            except:
                pass
            with open(join(segdir, 'targets.txt'), 'w') as f:
                for cortex in ['LPFC', 'LTC', 'MPFC', 'MTC', 'OCC', 'OFC', 'PC', 'SMC']:
                    f.write(join(self.subject_dir, 'ROI', '{}_{}.nii.gz\n'.format(sside, cortex)))


            # get most efficient gpu number
            if GPUtil.getGPUs()[gpu_num].load == 0:
                pass
            else:
                deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.1, maxMemory=0.5, attempts=1, interval=1900, verbose=False)
                gpu_num = deviceID[0]

            command = 'CUDA_VISIBLE_DEVICES={gpu_num} /usr/local/fsl/bin/probtrackx2_gpu \
            -x {roidir}/{sside}_thalamus.nii.gz \
            -l \
            --onewaycondition \
            -c 0.2 \
            -S 2000 \
            --steplength=0.5 \
            -P 5000 \
            --fibthresh=0.01 \
            --distthresh=0.0 \
            --sampvox=0.0 \
            --forcedir \
            --opd \
            -s {bedpostdir}/merged \
            -m {bedpostdir}/nodif_brain_mask \
            --xfm={t12dti_warp_mat} \
            --invxfm={t12dti_warp_mat_inv} \
            --dir={segdir} \
            --targetmasks={segdir}/targets.txt \
            --os2t'.format(gpu_num=gpu_num,
                           roidir=self.roi_dir, 
                           sside=sside,
                           bedpostdir=self.bedpostx_dir_unwarp,
                           t12dti_warp_mat=self.t1_to_dti_warp_and_mat,
                           t12dti_warp_mat_inv =self.t1_to_dti_warp_and_mat_inv,
                           segdir=segdir,
                           side=side)
            print(command)
            os.popen(command).read()
            
            command = 'find_the_biggest {}* {}/biggest.nii.gz'.format(
                join(segdir, 'seed'), segdir)
            print(command)
            os.popen(command).read()
            
    def thalamus_segmentation_check(self):
        check_roi_overlays_bilateral([self.t1_brain, 
                            join(self.seg_dir,'left/biggest.nii.gz'),
                            join(self.seg_dir,'right/biggest.nii.gz')])
        
class subjectWithFsDti(tcTractSettings):
    def __init__(self, subject_dir, gpu_num=0):
        '''
        Inheritance
        ---
        - subjectWithFsDti
            - tcTractSettings
                - antsSynSettings
                    - dtiEddyBedp
                        - dtiSettings
                            - mniSettings
                            - psyscanSettings
                    - fsSettings
                        - mniSettings
                        - psyscanSettings

        Functions run as __init__
        ---
        - invert_t1
        - t1_to_dti_registration
        - FS/mri/brain.mgz FS/mri/T1.mgz to nii
        '''
        dtiSettings.__init__(self, subject_dir)
        super().__init__(subject_dir)

    def run_dti_unwarp(self):
        if not isfile(self.dwi_data_eddy_out):         
            self.run_eddy()

        if not isfile(self.t1_brain_recip):
            invert_t1(self)

        if not isfile(self.t1_to_dti_flirt_mat_c3d):
            self.t1_to_dti_registration_c3d()

        if not isfile(self.t1_to_dti_ants_warp):
            self.ants_unwarping()

        if not isfile(self.dwi_data_unwarpped):
            self.apply_ants_unwarping()

        if not isfile(self.FA_unwarpped):
            self.dtifit_unwarpped()
        #self.ants_unwarping_check()

        # self.check_motion()
        if not isfile(self.t1_to_mni_fnirt):
            self.t1_to_mni_registration()
            self.dti_to_mni_through_t1_warp()

    def run_others_gpu(self):
        if not isfile(join(self.bedpostx_dir_unwarp,
                           'merged_th3samples.nii.gz')):
            self.bedpostx_gpu_unwarpped()
        #self.t1_to_dti_registration_check()
    #self.thalamus_segmentation_check()
        if not isfile(join(self.roi_dir, 
                           'lh_thalamus.nii.gz')):
            self.extract_rois()
        # self.extracted_roi_check()
        # self.extracted_roi_check_specific('lh_thalamus.nii.gz')
        if not isfile(self.t1_to_dti_warp_and_mat):
            self.make_seed_space_to_DTI_warps()

        if not isfile(join(self.seg_dir, 
                      'left', 
                      'fdt_path.nii.gz')):
            self.thalamo_cortical_tractography()

        # self.check_FA_registration()
        # self.check_label_overlay()
        # self.extract_bundle_FAs()
        # self.JHU_FA_df = df_tmp
        # self.make_FA_data_frame()
        # self.plot_JHU_FA()
        # self.plot_single_jhu('tract', 'Left', 'SLF')
        #if not all([isfile(x) for x in [self.t1_to_mni_flirt_mat,
                                        #self.t1_to_mni_fnirt,
                                        #self.t1_to_mni_fnirt_img]]):
            #self.t1_to_mni_registration(self)
    
    def t1_to_dti_registration_check(self):
        check_two_images(self.t1_to_dti_flirt_out, self.nodif)

class dtiOnlySubject(dtiSettings):
    def __init__(self, subject_dir):
        super().__init__(subject_dir)

    def MNI_registration(self):   
        # use T1 image to MNI space : FLIRT + FNIRT
        fsl_reg_command = 'fsl_reg {img_in} {ref} {img_out} -FA'.format(
            img_in=self.FA,
            ref=self.mni_fa_1mm,
            img_out=self.FA_map_MNI_name)
        os.popen(fsl_reg_command).read()

class psyscanStudy:
    def __init__(self, data_loc):
        self.data_loc = data_loc
        self.all_subject_locs = [join(data_loc, x) for x in os.listdir(data_loc) if x.startswith('PSY')]
        #self.all_subject_with_DTI_and_FS_locs = [x for x in all_subject_locs if 'DTI' in os.listdir(x) and 'FREESURFER' in os.listdir(x)]
        
        subjects_df = pd.DataFrame({'subject_loc':self.all_subject_locs})
        subjects_df.subject_class = subjects_df.subject_loc.apply(psyscanSettings)
        subjects_df['subject'] = subjects_df.subject_loc.apply(basename)
        subjects_df['group'] = subjects_df.subject_class.apply(lambda x: x.group)
        subjects_df['site'] = subjects_df.subject_class.apply(lambda x: x.site)
        subjects_df['site_name'] = subjects_df.subject_class.apply(lambda x: x.site_name)
        subjects_df['fs_data'] = subjects_df.subject_loc.apply(lambda x: True if 'FREESURFER' in os.listdir(x) else False)
        subjects_df['dti_data'] = subjects_df.subject_loc.apply(lambda x: True if 'DTI' in os.listdir(x) else False)
        subjects_df['both_data'] = subjects_df.fs_data & subjects_df.dti_data
        #for dti_img in ['
        #print(subjects_df[subjects_df.both_data].groupby('group').subject.count())
        #print(subjects_df.groupby(['group', 'both_data']).count())


if __name__ == "__main__":
    #data_loc = '/Volumes/CCNC_4T/psyscan/data'
    #psyscan_study = psyscanStudy(data_loc)

    dti_loc = '/Volumes/CCNC_4T/psyscan/data/PSYC15002/DTI/DTI.nii.gz'
    dti_img = nb.load(dti_loc)
    dti_data = dti_img.get_data()

    #plot_4d_dwi_pdf(dti_loc, '/home/kangik/prac.pdf')


    dti_loc = '/Volumes/CCNC_4T/psyscan/data/PSYC15002/DTI/nodif.nii.gz'
    plot_3d_dwi_pdf(dti_loc, '/home/kangik/prac.pdf')

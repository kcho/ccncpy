from __future__ import print_function
import progressbar
import os
from os.path import join, abspath, basename, dirname
import matplotlib.pyplot as plt

import argparse
import textwrap
import pandas as pd
import numpy as np
import nibabel as nb
import re

'''
Dependency : imagemagick

sudo apt-get install imagemagick
convert -delay 20 -loop 0 *.jpg myimage.gif
'''
mycmap = plt.cm.get_cmap('jet')
mycmap.set_under('magenta', alpha=0)

def get_biggest_loc_fs(subject, side):
    return abspath(join(dataLoc, subject, 'segmentation', side, 'biggest.nii.gz'))

def get_bg_T1_fs(subject):
    return abspath(join(dataLoc, subject, 'freesurfer/mri/brain.nii.gz'))

def get_map(imgLoc):
    return nb.load(imgLoc).get_data()

#def show_segmentation(subject):

def bbox2_3D(img):
    '''
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    '''
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def connectImgs(imgList, outgif):
    command = 'convert \
            -delay 10 \
            -loop 0 \
            {imgList} {outgif}'.format(
                imgList = ' '.join(imgList),
                outgif = outgif)

    command = re.sub('\s+', ' ', command)
    print(os.popen(command).read())
    for img in imgList:
        os.remove(img)
    
def makeGraph(bgMap, maskMap, mycmap, orientation):
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(maskMap)
    print('Max Nums :', rmin, rmax, cmin, cmax, zmin, zmax)

    fig, axes = plt.subplots(1, figsize=(5,5))


    imgList = []
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    if args.orientation=='x':
        ha = len(range(rmin, rmax))
        bar = progressbar.ProgressBar(widgets=widgets, maxval=ha).start()
        for num, sliceNum in enumerate(range(rmin, rmax)):
            axes.imshow(np.flipud(bgMap[sliceNum,:,:].T), 
                        cmap='gray',
                        extent=(0,1,1,0))
            axes.imshow(np.flipud(maskMap[sliceNum,:,:].T), 
                        cmap=mycmap, 
                        vmin=0.1, alpha=0.4,
                        extent=(0,1,1,0))
            axes.grid(False)
            name = '{0}.png'.format(sliceNum)
            axes.axis('tight')
            axes.axis('off')
            plt.savefig(name, bbox_inches='tight')
            imgList.append(name)
            bar.update(num)
        bar.finish()

    elif args.orientation=='y':
        ha = len(range(cmin, cmax))
        bar = progressbar.ProgressBar(widgets=widgets, maxval=ha).start()
        for num, sliceNum in enumerate(range(cmin, cmax)):
            axes.imshow(np.flipud(bgMap[:,sliceNum,:].T), 
                        cmap='gray',
                        extent=(0,1,1,0))
            axes.imshow(np.flipud(maskMap[:,sliceNum,:].T), 
                        cmap=mycmap, 
                        vmin=0.1, alpha=0.4,
                        extent=(0,1,1,0))
            axes.grid(False)
            name = '{0}.png'.format(sliceNum)
            axes.axis('tight')
            axes.axis('off')
            plt.savefig(name, bbox_inches='tight')
            imgList.append(name)
            bar.update(num)
        bar.finish()
    else:
        ha = len(range(zmin, zmax))
        bar = progressbar.ProgressBar(widgets=widgets, maxval=ha).start()
        for num, sliceNum in enumerate(range(zmin, zmax)):
            axes.imshow(np.flipud(bgMap[:,:,sliceNum].T), 
                        cmap='gray',
                        extent=(0,1,1,0))
            axes.imshow(np.flipud(maskMap[:,:,sliceNum].T), 
                        cmap=mycmap, 
                        vmin=0.1, alpha=0.4,
                        extent=(0,1,1,0))
            axes.grid(False)
            name = '{0}.png'.format(sliceNum)
            axes.axis('tight')
            axes.axis('off')
            plt.savefig(name, bbox_inches='tight')
            imgList.append(name)
            bar.update(num)
        bar.finish()

    return imgList


def makeGif(args):
    print(args)

    bgMap = get_map(args.bgImg)

    maskMaps = np.zeros_like(bgMap)
    for maskLoc in args.maskImg:
        maskMap = get_map(maskLoc)
        maskMaps = maskMaps + maskMap

    imgList = makeGraph(bgMap, maskMaps, mycmap, args.orientation)
    connectImgs(imgList, args.output)

    ##axes.imshow(np.flipud(leftSeg[:,:,maxSlice].T), cmap = mycmap, vmin=0.1, alpha=0.4)
    ##axes.imshow(np.flipud(rightSeg[:,:,maxSlice].T), cmap = mycmap, vmin=0.1, alpha=0.4)
    #axes.set_title(subject)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : Makes gif image from nifti files
            ========================================
            eg) {codeName} -b t1.nii.gz -m mask.nii.gz -o prac.gif
                Overlay mask.nii.gz on t1.nii.gz
            '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument(
        '-b', '--bgImg',
        help='Background nifti image')
    parser.add_argument(
        '-m', '--maskImg',
        nargs='+',
        help='Mask nifti image')
    parser.add_argument(
        '-o', '--output',
        help='Output gif name',
        default=abspath('out.gif'))
    parser.add_argument(
        '-p', '--orientation',
        help='Orientation. x, y, or z',
        default='z')
    parser.add_argument(
        '-r', '--range',
        help='Range of slices numbers to show',
        nargs=2, default=[0, 0], type=int)
    args = parser.parse_args()

    makeGif(args)

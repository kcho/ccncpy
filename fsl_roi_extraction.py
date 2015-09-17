#!/ccnc_bin/venv/bin/python

import pandas
import os
import re
import argparse
import sys
import textwrap

def get_roi_num_name(templateXml, templageImg):
    with open(templateXml,'r') as f:
        rawLines = f.readlines()
        
    nameLines = [x for x in rawLines if x.startswith('<label index')]
    nums_roi = [re.search('<label index="(\d+)',x).group(1) for x in nameLines]
    names_roi = [y.split('<')[0] for y in [x.split('>')[1] for x in nameLines]]
    names_roi = [re.sub(' ','_',x) for x in names_roi]
    names_roi = [re.sub(',','_',x) for x in names_roi]
    names_roi = [re.sub('\(','_',x) for x in names_roi]
    names_roi = [re.sub('\)','_',x) for x in names_roi]
    names_roi = [re.sub('__','_',x) for x in names_roi]
    names_roi = [re.sub('_$','',x) for x in names_roi]
    
    for num,name in zip(num_name, names_roi):
        print int(num)+1,name
        print 'fslmaths {inputImg} -thr {thr} -uthr {thr} {outputImg}'.format(
                    inputImg = templageImg,
                    thr = int(num)+1,
                    outputImg = name.lower())
        os.popen('fslmaths {inputImg} -thr {thr} -uthr {thr} {outputImg}'.format(
                    inputImg = templageImg,
                    thr = int(num)+1,
                    outputImg = name.lower())).read()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : Search files with user defined extensions 
            ========================================
            eg) {codeName} -e 'dcm|ima' -i /Users/kevin/NOR04_CKI
                Search dicom files in /Users/kevin/NOR04_CKI
            eg) {codeName} -c -e 'dcm|ima' -i /Users/kevin/NOR04_CKI
                Count dicom files in each directory under input 
            '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument(
        '-x', '--xml',
        help='xml location')

    parser.add_argument(
        '-i', '--image',
        help='template image location')

    args = parser.parse_args()

    get_roi_num_name(args.xml, args.image)

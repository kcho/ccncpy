#!/ccnc_bin/venv/bin/python

import os
import re
import argparse
import sys
import textwrap
import pandas as pd

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#pd.set_option('display.height', 1000)

def extSearch(ext,location=os.getcwd()):
    out = []
    for root,dirs,files in os.walk(location):
        for sFile in files:
            name = sFile.split('.')[0]
            extension = '.'.join(sFile.split('.')[1:])
            if re.search(ext,extension,re.IGNORECASE):
                out.append(os.path.join(root,sFile))
    return out

def subDirSearch(targetDirs,location=os.getcwd()):
    out = []
    for root,dirs,files in os.walk(location):
        inList = [x for x in dirs if x in targetDirs]
        if len(inList) == len(targetDirs):
            out.append(root)

    if len(out) == 1:
        return out

    elif len(out) == 0:
        print 'There are no matching directory'
        return None
        
    else:
        print 'There are more than 2 directories matching description'
        print out
        return out


def countExt(ext,location=os.getcwd()):
    extLocList = extSearch(ext,location)
    rootC = [os.path.split(x)[0] for x in extLocList]
    rootU = set(rootC)
    countD = {}
    for root in rootU:
        shorter_root = root.split(location)[1]
        count = len([x for x in extLocList if root in x])
        countD[shorter_root] = count
    return countD
    
def dict2pd(D):
    return pd.DataFrame.from_dict(D,orient='index')

def list2pd(D):
    return pd.DataFrame.from_list(D,orient='index')

def main(args):
    if not args.count:
        for i in extSearch(args.extension,args.inputDir):
            print i
    else:
        print dict2pd(countExt(args.extension,args.inputDir))

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
        '-i', '--inputDir',
        help='Data directory location, default=pwd',
        default=os.getcwd())
    parser.add_argument(
        '-c', '--count',
        help='count files with the ext in each directory',
        action='store_true')
    parser.add_argument(
        '-e', '--extension',
        help='Extension to search')
    args = parser.parse_args()

    if not args.extension:
        parser.error('No extension given, add -e or --extension')

    main(args)

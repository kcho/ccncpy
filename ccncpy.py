#!/ccnc_bin/venv/bin/python

import os
import re
import pandas as pd


def extSearch(ext,location=os.getcwd()):
    out = []
    for root,dirs,files in os.walk(location):
        for sFile in files:
            name = sFile.split('.')[0]
            extention = '.'.join(sFile.split('.')[1:])
            if re.search(ext,extention,re.IGNORECASE):
                out.append(os.path.join(root,sFile))
    return out

def countDicom(ext,location=os.getcwd()):
    extLocList = extSearch(ext,location)
    rootC = [os.path.split(x)[0] for x in extLocList]
    rootU = set(rootC)
    countD = {}
    for root in rootU:
        count = len([x for x in extLocList if root in x])
        countD[root] = count
    return countD
    


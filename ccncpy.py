#!/ccnc_bin/venv/bin/python

import os
import re


def dirSearch(word,location=os.getcwd()):
    for root,dirs,files in os.walk(location):
        if re.search(word,location,re.IGNORECASE):
            print root

def extSearch(ext,location=os.getcwd()):
    for root,dirs,files in os.walk(location):
        for sFile in files:
            name = sFile.split('.')[0]
            extention = sFile.split('.')[1:]

            if re.search(ext,extention,re.IGNORECASE):
                print os.path.join(root,sFile)

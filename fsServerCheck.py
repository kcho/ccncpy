import re
import sys
from os.path import join
import os
import pandas as pd

def main(dirLoc):
    subjectList = [x for x in os.listdir(dirLoc) if re.search('(FEP)|(CHR)|(NOR)', x)]

    serverList = pd.DataFrame()
    for subject in subjectList:
        fsLog = join(dirLoc, subject, 'FREESURFER/scripts/recon-all.log')
        with open(fsLog, 'r') as f:
            server = f.readlines()[9].split(' ')[0:2]
            subj_dict = {subject:{'server':' '.join(server)}}
            serverList = pd.concat([serverList, pd.DataFrame.from_dict(subj_dict, orient='index')])

    serverList = serverList.reset_index()
    serverList.columns = ['subject', 'server']
    serverList['group'] = serverList['subject'].str[:3]
    print(serverList.reset_index().groupby(['group', 'server']).count())

if __name__=='__main__':
    main(sys.argv[1])

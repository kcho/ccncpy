import re
import os
from os.path import basename, join, dirname, abspath, isfile
import pandas as pd

def main():
    subjects = [x for x in os.listdir(os.getcwd()) if x.startswith('FEP') or x.startswith('CHR') or x.startswith('NOR')]
    df = pd.DataFrame()
    df['subject'] = subjects
    df['group'] = df.subject.str[:3]
    df['logLoc'] = df.subject + '/FREESURFER/scripts/recon-all.log'
    df['check'] = df.logLoc.apply(isfile).map({True:'finished', False:'not finished'})

    df = df.sort_values('subject').reset_index()

    print df.groupby(['group','check']).count()['subject']

if __name__=='__main__':
    main()
#for i in [FNC]*
#do
    #if [ -e ${i}/FREESURFER/scripts/recon-all.log ]
    #then
        #echo ${i} `cat ${i}/FREESURFER/scripts/recon-all.log| tail -n 1`
    #else
        #echo ${i} no freesurfer yet
    #fi
#done

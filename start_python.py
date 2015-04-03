import os
import shutil

def main():
    script=os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.py')
    print script
    newScript = raw_input('Name of the script : ')
    shutil.copy(script,os.path.join(os.getcwd(),newScript))

if __name__=='__main__':
    main()

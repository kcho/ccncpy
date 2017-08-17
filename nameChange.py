import mriName
import sys


def nameChange2old(name):
    invMap = {v: k for k, v in mriName.folderNameDict.iteritems()}
    f = lambda x: invMap[x] if x in invMap.keys() else x
    print(f(name))

if __name__ == '__main__':
    nameChange(sys.argv[1])

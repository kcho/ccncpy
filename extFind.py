#!/ccnc_bin/venv/bin/python
from ccncpy import *

def main(args):
    if not args.count:
        for i in extSearch(args.extension,args.inputDir):
            print i
    else:
        df = dict2pd(countExt(args.extension,args.inputDir))
        if args.output.endswith('xls'):
            df.to_excel(args.output)
        elif args.output.endswith('txt'):
            df.to_csv(args.output,sep='\t')
        elif args.output.endswith('csv'):
            df.to_csv(args.output)
        else:
            print df
        

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
        '-o', '--output',
        default='print',
        help='Eg. output.xlsx. Output types, [xls,csv,txt]. If not specified, just prints')
    parser.add_argument(
        '-e', '--extension',
        help='Extension to search')
    args = parser.parse_args()

    if not args.extension:
        parser.error('No extension given, add -e or --extension')

    main(args)

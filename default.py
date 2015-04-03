#!/ccnc_bin/venv/bin/python


import os
import re
import argparse
import textwrap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : 
            ========================================
            eg) {codeName} --input {in_put} --output {output}
            '''.format(codeName=os.path.basename(__file__),
                       in_put = 'haha',
                       output = 'hoho')))

    parser.add_argument(
        '-i', '--input',
        help='Input',
        default=os.getcwd())

    parser.add_argument(
        '-o', '--output',
        help='Output',
        default=os.getcwd())

    if not args.extension:
        parser.error('No extension given, add -e or --extension')

    main(args)

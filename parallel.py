#!/ccnc_bin/venv/bin/python

import os
import re
import argparse
import textwrap
import pp

def main(args):

    port = '35000'
    serverSecret = 'ccncserver'
    ncpus = 20
    ppservers=tuple([x+':'+port for x in args.server])

    #job_server = pp.Server(ppservers=ppservers, secret="mysecret") 
    # tuple of all parallel python servers to connect with
    #ppservers = ("*",)
    #ppservers = ("10.0.0.1",)
    #if len(sys.argv) > 1:
    #ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers, secret=serverSecret)
    #else:
    # Creates jobserver with automatically detected number of workers
    #job_server = pp.Server(ppservers=ppservers, secret="ccncserver")

    print "Starting pp with", job_server.get_ncpus(), "workers"

    start_time = time.time()

    # The following submits 8 jobs and then retrieves the results
    #commandFile = '/Volumes/CCNC_3T_2/kcho/script.sh'

    #with open(commandFile,'r') as f:
        #commandsToRun = f.readlines()

    jobs = [(command, 
             job_server.submit(jobDispatch,
                               (command,), 
                               () , 
                               ("os","sys","re","argparse","textwrap",))) for command in commandsToRun]

    for command, job in jobs:
        print command, "is completed", job()

    print "Time elapsed: ", time.time() - start_time, "s"
    job_server.print_stats()


def jobDispatch(job):
    output = os.popen(job).read()
    return output


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
        '-s', '--server',
        help='Server list')

    args = parser.parse_args()

    if not args.server:
        parser.error('No server list given, add -s or --server')

    main(args)

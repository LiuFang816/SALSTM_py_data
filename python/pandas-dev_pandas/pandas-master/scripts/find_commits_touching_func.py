#!/usr/bin/env python
# -*- coding: utf-8 -*-

# copryright 2013, y-p @ github

from __future__ import print_function
from pandas.compat import range, lrange, map

"""Search the git history for all commits touching a named method

You need the sh module to run this
WARNING: this script uses git clean -f, running it on a repo with untracked files
will probably erase them.
"""
import logging
import re
import os
from collections import namedtuple
from pandas.compat import parse_date

try:
    import sh
except ImportError:
    raise ImportError("The 'sh' package is required in order to run this script. ")

import argparse

desc = """
Find all commits touching a sepcified function across the codebase.
""".strip()
argparser = argparse.ArgumentParser(description=desc)
argparser.add_argument('funcname', metavar='FUNCNAME',
                   help='Name of function/method to search for changes on.')
argparser.add_argument('-f', '--file-masks', metavar='f_re(,f_re)*',
                       default=["\.py.?$"],
                   help='comma seperated list of regexes to match filenames against\n'+
                   'defaults all .py? files')
argparser.add_argument('-d', '--dir-masks', metavar='d_re(,d_re)*',
                       default=[],
                   help='comma seperated list of regexes to match base path against')
argparser.add_argument('-p', '--path-masks', metavar='p_re(,p_re)*',
                       default=[],
                   help='comma seperated list of regexes to match full file path against')
argparser.add_argument('-y', '--saw-the-warning',
                       action='store_true',default=False,
                   help='must specify this to run, acknowledge you realize this will erase untracked files')
argparser.add_argument('--debug-level',
                       default="CRITICAL",
                   help='debug level of messages (DEBUG,INFO,etc...)')

args = argparser.parse_args()


lfmt = logging.Formatter(fmt='%(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

shh = logging.StreamHandler()
shh.setFormatter(lfmt)

logger=logging.getLogger("findit")
logger.addHandler(shh)


Hit=namedtuple("Hit","commit path")
HASH_LEN=8

def clean_checkout(comm):
    h,s,d = get_commit_vitals(comm)
    if len(s) > 60:
        s = s[:60] + "..."
    s=s.split("\n")[0]
    logger.info("CO: %s %s" % (comm,s ))

    sh.git('checkout', comm ,_tty_out=False)
    sh.git('clean', '-f')

def get_hits(defname,files=()):
    cs=set()
    for f in files:
        try:
            r=sh.git('blame', '-L', '/def\s*{start}/,/def/'.format(start=defname),f,_tty_out=False)
        except sh.ErrorReturnCode_128:
            logger.debug("no matches in %s" % f)
            continue

        lines = r.strip().splitlines()[:-1]
        # remove comment lines
        lines = [x for x in lines if not re.search("^\w+\s*\(.+\)\s*#",x)]
        hits = set(map(lambda x: x.split(" ")[0],lines))
        cs.update(set([Hit(commit=c,path=f) for c in hits]))

    return cs

def get_commit_info(c,fmt,sep='\t'):
    r=sh.git('log', "--format={}".format(fmt), '{}^..{}'.format(c,c),"-n","1",_tty_out=False)
    return compat.text_type(r).split(sep)

def get_commit_vitals(c,hlen=HASH_LEN):
    h,s,d= get_commit_info(c,'%H\t%s\t%ci',"\t")
    return h[:hlen],s,parse_date(d)

def file_filter(state,dirname,fnames):
    if args.dir_masks and not any([re.search(x,dirname) for x in args.dir_masks]):
        return
    for f in fnames:
        p = os.path.abspath(os.path.join(os.path.realpath(dirname),f))
        if  any([re.search(x,f) for x in args.file_masks])\
            or any([re.search(x,p) for x in args.path_masks]):
            if os.path.isfile(p):
                state['files'].append(p)

def search(defname,head_commit="HEAD"):
    HEAD,s = get_commit_vitals("HEAD")[:2]
    logger.info("HEAD at %s: %s" % (HEAD,s))
    done_commits = set()
    # allhits = set()
    files = []
    state = dict(files=files)
    os.path.walk('.',file_filter,state)
    # files now holds a list of paths to files

    # seed with hits from q
    allhits= set(get_hits(defname, files = files))
    q = set([HEAD])
    try:
        while q:
            h=q.pop()
            clean_checkout(h)
            hits = get_hits(defname, files = files)
            for x in hits:
                prevc = get_commit_vitals(x.commit+"^")[0]
                if prevc not in done_commits:
                    q.add(prevc)
            allhits.update(hits)
            done_commits.add(h)

            logger.debug("Remaining: %s" % q)
    finally:
        logger.info("Restoring HEAD to %s" % HEAD)
        clean_checkout(HEAD)
    return allhits

def pprint_hits(hits):
    SUBJ_LEN=50
    PATH_LEN = 20
    hits=list(hits)
    max_p = 0
    for hit in hits:
        p=hit.path.split(os.path.realpath(os.curdir)+os.path.sep)[-1]
        max_p=max(max_p,len(p))

    if max_p < PATH_LEN:
        SUBJ_LEN += PATH_LEN - max_p
        PATH_LEN = max_p

    def sorter(i):
        h,s,d=get_commit_vitals(hits[i].commit)
        return hits[i].path,d

    print("\nThese commits touched the %s method in these files on these dates:\n" \
          % args.funcname)
    for i in sorted(lrange(len(hits)),key=sorter):
        hit = hits[i]
        h,s,d=get_commit_vitals(hit.commit)
        p=hit.path.split(os.path.realpath(os.curdir)+os.path.sep)[-1]

        fmt = "{:%d} {:10} {:<%d} {:<%d}" % (HASH_LEN, SUBJ_LEN, PATH_LEN)
        if len(s) > SUBJ_LEN:
            s = s[:SUBJ_LEN-5] + " ..."
        print(fmt.format(h[:HASH_LEN],d.isoformat()[:10],s,p[-20:]) )

    print("\n")

def main():
    if not args.saw_the_warning:
        argparser.print_help()
        print("""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: this script uses git clean -f, running it on a repo with untracked files.
It's recommended that you make a fresh clone and run from its root directory.
You must specify the -y argument to ignore this warning.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""")
        return
    if isinstance(args.file_masks,compat.string_types):
        args.file_masks = args.file_masks.split(',')
    if isinstance(args.path_masks,compat.string_types):
        args.path_masks = args.path_masks.split(',')
    if isinstance(args.dir_masks,compat.string_types):
        args.dir_masks = args.dir_masks.split(',')

    logger.setLevel(getattr(logging,args.debug_level))

    hits=search(args.funcname)
    pprint_hits(hits)

    pass

if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/python
"""
This script requires to obtain a username and password to obtain a copy of the MPII-DVS data set.
See here: http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/mpii-movie-description-dataset/

Please run it in place, and it will create the story/dvs content, and also link movies.json to appropriate DVS files
"""

import os
import sys
import ipdb
import json
import shutil
import requests
import subprocess


### Check that two arguments exist, otherwise exit
if len(sys.argv) < 3:
    print 'Usage:'
    print 'python download_dvs_stories.py <mpi-username> <mpi-password>'
    sys.exit()

username = sys.argv[1]
password = sys.argv[2]


### Download the required DVS data file
print '************* MovieQA *************'
print 'Downloading DVS stories'
print '***********************************'

print '>>> Downloading DVS transcriptions'

DVS_DATA = 'annotations-original.csv'
DVS_URL = 'http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/' + DVS_DATA

r = requests.get(DVS_URL, auth=(username, password))
if r.status_code == requests.codes.ok:
    with open(DVS_DATA, 'wb') as out:
        for bits in r.iter_content():
            out.write(bits)
else:
    r.raise_for_status()


### Burst the CSV into separate files
print '>>> Creating new DVS story files'
with open(DVS_DATA) as fid:
    dvs_lines = fid.readlines()
    dvs_lines = [d.strip() for d in dvs_lines]

# get movies which are part of the list
MOVIES_JSON = 'data/movies.json'
with open(MOVIES_JSON, 'r') as fid:
    movies = json.load(fid)
movies_list_imdbkeys = [m['imdb_key'] for m in movies]

dvs_imdbkey_mappings = \
        {'0001': 'tt0169547', '0002': 'tt0119822', '0003': 'tt0034583', '0004': 'tt0056923',
         '0005': 'tt0071315', '0006': 'tt0109445', '0008': 'tt0116282', '0009': 'tt0109830',
         '0011': 'tt0083987', '0012': 'tt0113161', '0013': 'tt0077651', '0016': 'tt0190590',
         '0017': 'tt0253474', '0019': 'tt0110912', '0020': 'tt0093822', '0021': 'tt0047396',
         '0022': 'tt0105236', '0023': 'tt0289879', '0024': 'tt0120737', '0025': 'tt0167260',
         '0026': 'tt0319061', '0027': 'tt0118715', '0028': 'tt0104036', '0029': 'tt0061722',
         '0030': 'tt0054997', '0031': 'tt0037884', '0032': 'tt0093779', '0033': 'tt0086879',
         '0038': 'tt0054215', '0041': 'tt0167404', '0043': 'tt0103074', '0046': 'tt0118842',
         '0049': 'tt0091167', '0050': 'tt0097576', '0051': 'tt0119654', '0053': 'tt0119643',
         '1001': 'tt1907668', '1002': 'tt0417741', '1003': 'tt0455538', '1006': 'tt1010048',
         '1007': 'tt0145487', '1008': 'tt0316654', '1009': 'tt0413300', '1010': 'tt0120338',
         '1011': 'tt1454029', '1014': 'tt1190080', '1015': 'tt0988595', '1018': 'tt0758774',
         '1019': 'tt1093908', '1020': 'tt1570728', '1023': 'tt1499658', '1024': 'tt2024432',
         '1028': 'tt0481141', '1030': 'tt1152836', '1031': 'tt0830515', '1033': 'tt1515091',
         '1035': 'tt1385826', '1037': 'tt0421715', '1038': 'tt1343092', '1039': 'tt0436697',
         '1042': 'tt1193138', '1043': 'tt0443274', '1045': 'tt1174732', '1046': 'tt0455824',
         '1047': 'tt1034303', '1048': 'tt1205489', '1049': 'tt0295297', '1050': 'tt0926084',
         '1052': 'tt0373889', '1053': 'tt0241527', '1055': 'tt0822832', '1056': 'tt0462499',
         '1057': 'tt0814314', '1058': 'tt1226271', '1059': 'tt0458352', '1060': 'tt1068680',
         '1061': 'tt1201607', '1062': 'tt0970416', '2004': 'tt0467406', '2005': 'tt0286106',
         '2012': 'tt0217869', '2017': 'tt0307987', '2026': 'tt1038686', '2027': 'tt1707386',
         '2034': 'tt1650062', '2040': 'tt1142988', '2041': 'tt1758830', '2051': 'tt0330373',
         '2054': 'tt0304141'}

STORY_DIR = 'story/dvs'
if not os.path.isdir(STORY_DIR):
    os.makedirs(STORY_DIR)


### Create individual DVS files (like .srt)
for dvskey, imdbkey in dvs_imdbkey_mappings.iteritems():
    if imdbkey not in movies_list_imdbkeys:
        continue

    # show which movie is currently updating 
    this_dvs_lines = [d for d in dvs_lines if d.startswith(dvskey)]
    print '++ Adding DVS for {} -- {} -- {}'.format(dvskey, imdbkey, movies[movies_list_imdbkeys.index(imdbkey)]['name'])

    destination = os.path.join(STORY_DIR, imdbkey + '.dvs')

    # write text to "srt" like file
    fid = open(destination, 'w')
    for k, line in enumerate(this_dvs_lines):
        meta, text = line.split('\t')
        timestamps = meta[-25:]
        [h1, m1, s1, ms1], [h2, m2, s2, ms2] = [ts.split('.') for ts in timestamps.split('-')]

        fid.write('%d\n' %(k+1))
        fid.write('%s:%s:%s,%s --> %s:%s:%s,%s\n' %(h1, m1, s1, ms1, h2, m2, s2, ms2))
        fid.write('%s\n\n' %text)
    fid.close()


### Update movies.json
print '>>> Updating movies.json to link to DVS files'

# add link to the DVS in the movies.json
for dvskey, imdbkey in dvs_imdbkey_mappings.iteritems():
    if imdbkey not in movies_list_imdbkeys:
        continue
    idx = movies_list_imdbkeys.index(imdbkey)
    dvs_file = os.path.join(STORY_DIR, imdbkey + '.dvs')
    assert os.path.exists(dvs_file), 'DVS file for %s -- %s not found!' %(dvskey, imdbkey)
    movies[idx]['text']['dvs'] = dvs_file

with open(MOVIES_JSON, 'w') as outfile:
    json.dump(movies, outfile, indent=2)


# X files have been added
added_nfiles = len(os.listdir(STORY_DIR))
print '>>> Enabled DVS-based answering for %d movies' %added_nfiles
assert added_nfiles == 60, 'DVS based QA should be enabled for 60 movies'


### Cleanup
os.remove(DVS_DATA)



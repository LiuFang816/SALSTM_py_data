# coding=utf-8

import os
import sys
import unittest

from zhihu_oauth import ZhihuClient


TOKEN_FILE_NAME = 'token.pkl.' + str(sys.version_info.major)


class ZhihuClientClassTest(unittest.TestCase):
    def setUp(self):
        super(ZhihuClientClassTest, self).setUp()

        if not os.path.isdir('test') and os.path.isfile(TOKEN_FILE_NAME):
            os.chdir('..')

        token_file_path = os.path.join('test', TOKEN_FILE_NAME)

        if not os.path.isfile(token_file_path):
            print('\nno token file, skip all tests.')
            self.skipTest('no token file.')

        self.client = ZhihuClient()

        try:
            self.client.load_token(token_file_path)
        except ValueError:
            print(
                '\ntoken version not math python version, skip all tests.')
            self.skipTest('token version not math python version.')

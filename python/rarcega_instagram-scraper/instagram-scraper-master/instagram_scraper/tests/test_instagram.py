import unittest
import os
import shutil
import tempfile
import requests_mock
from instagram_scraper import InstagramScraper
from instagram_scraper.constants import *

class InstagramTests(unittest.TestCase):

    def setUp(self):
        fixtures_path = os.path.join(os.path.dirname(__file__), 'fixtures')
        self.response_user_metadata = open(os.path.join(fixtures_path,
                                                        'response_user_metadata.json')).read()
        self.response_first_page = open(os.path.join(fixtures_path,
                                                     'response_first_page.json')).read()
        self.response_second_page = open(os.path.join(fixtures_path,
                                                      'response_second_page.json')).read()

        self.test_dir = tempfile.mkdtemp()

        # This is a max id of the last item in response_first_page.json.
        self.max_id = "1369793132326237681_50955533"

        self.scraper = InstagramScraper("test", dst=self.test_dir, quiet=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_scrape(self):
        with requests_mock.Mocker() as m:
            m.get(BASE_URL + self.scraper.usernames[0], text=self.response_user_metadata)
            m.get(MEDIA_URL.format(self.scraper.usernames[0]), text=self.response_first_page)
            m.get(MEDIA_URL.format(self.scraper.usernames[0]) + '?max_id=' + self.max_id,
                  text=self.response_second_page)
            m.get('https://fake-url.com/photo1.jpg', text="image1")
            m.get('https://fake-url.com/photo2.jpg', text="image2")
            m.get('https://fake-url.com/photo3.jpg', text="image3")

            self.scraper.scrape()

            # First page has photo1 and photo2, while second page has photo3. If photo3
            # is opened, generator successfully traversed both pages.
            self.assertEqual(open(os.path.join(self.test_dir, 'photo3.jpg')).read(),
                             "image3")




#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import errno
import json
import logging.config
import os
import re
import time
import warnings

import concurrent.futures
import requests
import tqdm

from instagram_scraper.constants import *

warnings.filterwarnings('ignore')

class InstagramScraper(object):

    """InstagramScraper scrapes and downloads an instagram user's photos and videos"""

    def __init__(self, usernames, login_user=None, login_pass=None, dst=None, quiet=False, max=0, retain_username=False):
        self.usernames = usernames if isinstance(usernames, list) else [usernames]
        self.login_user = login_user
        self.login_pass = login_pass
        self.max = max
        self.retain_username = retain_username
        self.dst = './' if dst is None else dst

        # Controls the graphical output of tqdm
        self.quiet = quiet

        # Set up a file logger.
        self.logger = InstagramScraper.get_logger(level=logging.DEBUG)

        self.session = requests.Session()
        self.cookies = None
        self.logged_in = False

        if self.login_user and self.login_pass:
            self.login()

    def login(self):
        """Logs in to instagram"""
        self.session.headers.update({'Referer': BASE_URL})
        req = self.session.get(BASE_URL)

        self.session.headers.update({'X-CSRFToken': req.cookies['csrftoken']})

        login_data = {'username': self.login_user, 'password': self.login_pass}
        login = self.session.post(LOGIN_URL, data=login_data, allow_redirects=True)
        self.session.headers.update({'X-CSRFToken': login.cookies['csrftoken']})
        self.cookies = login.cookies

        if login.status_code == 200 and json.loads(login.text)['authenticated']:
            self.logged_in = True
        else:
            self.logger.exception('Login failed for ' + self.login_user)
            raise ValueError('Login failed for ' + self.login_user)

    def logout(self):
        """Logs out of instagram"""
        if self.logged_in:
            try:
                logout_data = {'csrfmiddlewaretoken': self.cookies['csrftoken']}
                self.session.post(LOGOUT_URL, data=logout_data)
                self.logged_in = False
            except requests.exceptions.RequestException:
                self.logger.warning('Failed to log out ' + self.login_user)

    def make_dst_dir(self, username):
        '''Creates the destination directory'''
        if self.dst == './':
            dst = './' + username
        else:
            if self.retain_username:
                dst = self.dst + '/' + username
            else:
                dst = self.dst

        try:
            os.makedirs(dst)
        except OSError as err:
            if err.errno == errno.EEXIST and os.path.isdir(dst):
                # Directory already exists
                pass
            else:
                # Target dir exists as a file, or a different error
                raise

        return dst

    def scrape(self):
        """Crawls through and downloads user's media"""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        for username in self.usernames:
            future_to_item = {}

            # Make the destination dir.
            dst = self.make_dst_dir(username)

            # Get the user metadata.
            user = self.fetch_user(username)

            if user:
                # Download the profile pic if not the default.
                if 'profile_pic_url_hd' in user and '11906329_960233084022564_1448528159' not in user['profile_pic_url_hd']:
                    item = {'url': re.sub(r'/s\d{3,}x\d{3,}/', '/', user['profile_pic_url_hd'])}
                    for item in tqdm.tqdm([item], desc='Searching {0} for profile pic'.format(username), unit=" images", ncols=0, disable=self.quiet):
                        future = executor.submit(self.download, item, dst)
                        future_to_item[future] = item

                if self.logged_in:
                    # Get the user's stories.
                    stories = self.fetch_stories(user['id'])

                    # Downloads the user's stories and sends it to the executor.
                    iter = 0
                    for item in tqdm.tqdm(stories, desc='Searching {0} for stories'.format(username), unit=" media", disable=self.quiet):
                        iter = iter + 1
                        if ( self.max != 0 and iter >= self.max ):
                            break
                        else:
                            future = executor.submit(self.download, item, dst)
                            future_to_item[future] = item

            # Crawls the media and sends it to the executor.
            iter = 0
            for item in tqdm.tqdm(self.media_gen(username), desc='Searching {0} for posts'.format(username),
                                unit=' media', disable=self.quiet):
                iter = iter + 1
                if ( self.max != 0 and iter >= self.max ):
                    break
                else:
                    future = executor.submit(self.download, item, dst)
                    future_to_item[future] = item

            # Displays the progress bar of completed downloads. Might not even pop up if all media is downloaded while
            # the above loop finishes.
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item),
                                desc='Downloading', disable=self.quiet):
                item = future_to_item[future]

                if future.exception() is not None:
                    self.logger.warning('Media id {0} at {1} generated an exception: {2}'.format(item['id'], item['url'], future.exception()))

        self.logout()

    def fetch_user(self, username):
        """Fetches the user's metadata"""
        resp = self.session.get(BASE_URL + username)

        if resp.status_code == 200 and '_sharedData' in resp.text:
            try:
                shared_data = resp.text.split("window._sharedData = ")[1].split(";</script>")[0]
                return json.loads(shared_data)['entry_data']['ProfilePage'][0]['user']
            except (TypeError, KeyError, IndexError):
                pass

    def fetch_stories(self, user_id):
        """Fetches the user's stories"""
        resp = self.session.get(STORIES_URL.format(user_id), headers={
            'user-agent' : STORIES_UA,
            'cookie'     : STORIES_COOKIE.format(self.cookies['ds_user_id'], self.cookies['sessionid'])
        })

        retval = json.loads(resp.text)

        if resp.status_code == 200 and 'items' in retval and len(retval['items']) > 0:
            return [self.set_story_url(item) for item in retval['items']]
        return []

    def media_gen(self, username):
        """Generator of all user's media"""
        try:
            media = self.fetch_media_json(username, max_id=None)

            while True:
                for item in media['items']:
                    yield item
                if media.get('more_available'):
                    max_id = media['items'][-1]['id']
                    media = self.fetch_media_json(username, max_id)
                else:
                    return
        except ValueError:
            self.logger.exception('Failed to get media for ' + username)

    def fetch_media_json(self, username, max_id):
        """Fetches the user's media metadata"""
        url = MEDIA_URL.format(username)

        if max_id is not None:
            url += '?&max_id=' + max_id

        resp = self.session.get(url)

        if resp.status_code == 200:
            media = json.loads(resp.text)

            if not media['items']:
                raise ValueError('User {0} is private'.format(username))

            media['items'] = [self.set_media_url(item) for item in media['items']]
            return media
        else:
            raise ValueError('User {0} does not exist'.format(username))

    def set_media_url(self, item):
        """Sets the media url"""
        item['url'] = item[item['type'] + 's']['standard_resolution']['url'].split('?')[0]
        # remove dimensions to get largest image
        item['url'] = re.sub(r'/s\d{3,}x\d{3,}/', '/', item['url'])
        # get non-square image if one exists
        item['url'] = re.sub(r'/c\d{1,}.\d{1,}.\d{1,}.\d{1,}/', '/', item['url'])
        return item

    def set_story_url(self, item):
        """Sets the story url"""
        item['url'] = item['image_versions2']['candidates'][0]['url'].split('?')[0]
        return item

    def download(self, item, save_dir='./'):
        """Downloads the media file"""
        base_name = item['url'].split('/')[-1]
        file_path = os.path.join(save_dir, base_name)

        if not os.path.isfile(file_path):
            with open(file_path, 'wb') as media_file:
                try:
                    content = self.session.get(item['url']).content
                except requests.exceptions.ConnectionError:
                    time.sleep(5)
                    content = requests.get(item['url']).content

                media_file.write(content)

            file_time = int(item.get('created_time', item.get('taken_at', time.time())))
            os.utime(file_path, (file_time, file_time))

    @staticmethod
    def get_logger(level=logging.WARNING, log_file='instagram-scraper.log'):
        '''Returns a file logger.'''
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.NOTSET)

        handler = logging.FileHandler(log_file, 'w')
        handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    @staticmethod
    def parse_file_usernames(usernames_file):
        '''Parses a file containing a list of usernames.'''
        users = []

        try:
            with open(usernames_file) as user_file:
                for line in user_file.readlines():
                    # Find all usernames delimited by ,; or whitespace
                    users += re.findall(r'[^,;\s]+', line)
        except IOError as err:
            raise ValueError('File not found ' + err)

        return users

    @staticmethod
    def parse_str_usernames(usernames_str):
        '''Parse the username input as a delimited string of users.'''
        return re.findall(r'[^,;\s]+', usernames_str)

def main():
    max = 0
    parser = argparse.ArgumentParser(
        description="instagram-scraper scrapes and downloads an instagram user's photos and videos.")

    parser.add_argument('username', help='Instagram user(s) to scrape', nargs='*')
    parser.add_argument('--destination', '-d', help='Download destination')
    parser.add_argument('--login_user', '-u', help='Instagram login user')
    parser.add_argument('--login_pass', '-p', help='Instagram login password')
    parser.add_argument('--filename', '-f', help='Path to a file containing a list of users to scrape')
    parser.add_argument('--quiet', '-q', action='store_true', help='Be quiet while scraping')
    parser.add_argument('--maximum', '-m', type=int, default=0, help='Maximum number of items to scrape')
    parser.add_argument('--retain_username', '-n', action='store_true',
                        help='Creates username subdirectory when destination flag is set')

    args = parser.parse_args()

    if (args.login_user and args.login_pass is None) or (args.login_user is None and args.login_pass):
        parser.print_help()
        raise ValueError('Must provide login user AND password')

    if not args.username and args.filename is None:
        parser.print_help()
        raise ValueError('Must provide username(s) OR a file containing a list of username(s)')
    elif args.username and args.filename:
        parser.print_help()
        raise ValueError('Must provide only one of the following: username(s) OR a filename containing username(s)')
    usernames = []

    if args.filename:
        usernames = InstagramScraper.parse_file_usernames(args.filename)
    else:
        usernames = InstagramScraper.parse_str_usernames(','.join(args.username))

    scraper = InstagramScraper(usernames, args.login_user, args.login_pass, args.destination, args.quiet, args.maximum, args.retain_username)
    scraper.scrape()

if __name__ == '__main__':
    main()

BASE_URL = 'https://www.instagram.com/'
LOGIN_URL = BASE_URL + 'accounts/login/ajax/'
LOGOUT_URL = BASE_URL + 'accounts/logout/'
MEDIA_URL = BASE_URL + '{0}/media'

STORIES_URL = 'https://i.instagram.com/api/v1/feed/user/{0}/reel_media/'
STORIES_UA = 'Instagram 9.5.2 (iPhone7,2; iPhone OS 9_3_3; en_US; en-US; scale=2.00; 750x1334) AppleWebKit/420+'
STORIES_COOKIE = 'ds_user_id={0}; sessionid={1};'

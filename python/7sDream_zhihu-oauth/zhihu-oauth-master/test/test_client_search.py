# coding=utf-8

from .test_base_class import ZhihuClientClassTest

from zhihu_oauth.zhcls.search import SearchResult, SearchType
from zhihu_oauth.zhcls.people import People
from zhihu_oauth.zhcls.column import Column
from zhihu_oauth.zhcls.topic import Topic
from zhihu_oauth.zhcls.live import Live

GeniusQiQi = '7sDream'
MiaoMiaoColumn = '喵星陨石坑'
LoveTopic = '爱情'
IQLive = '一小时建立终生受用的阅读操作系统'
# Offline = '离线'


class TestZhihuClientSearch(ZhihuClientClassTest):
    def test_client_user_search(self):
        self.assertIsInstance(
            self.client.search(GeniusQiQi, SearchType.PEOPLE)[0], SearchResult
        )
        self.assertIsInstance(
            self.client.search(GeniusQiQi, SearchType.PEOPLE)[0].obj, People
        )

    def test_client_column_search(self):
        self.assertIsInstance(
            self.client.search(MiaoMiaoColumn, SearchType.COLUMN)[0],
            SearchResult
        )
        self.assertIsInstance(
            self.client.search(MiaoMiaoColumn, SearchType.COLUMN)[0].obj,
            Column
        )

    def test_client_topic_search(self):
        self.assertIsInstance(
            self.client.search(LoveTopic, SearchType.TOPIC)[0], SearchResult
        )
        self.assertIsInstance(
            self.client.search(LoveTopic, SearchType.TOPIC)[0].obj, Topic
        )

    def test_client_live_search(self):
        self.assertIsInstance(
            self.client.search(IQLive, SearchType.LIVE)[0], SearchResult
        )
        self.assertIsInstance(
            self.client.search(IQLive, SearchType.LIVE)[0].obj, Live
        )

    # coming soon!
    #
    # def test_client_ebook_search(self):
    #     self.assertIsInstance(
    #         self.client.search(Offline, SearchType.EBOOK)[0], SearchResult
    #     )
    #     self.assertIsInstance(
    #         self.client.search(Offline, SearchType.EBOOK)[0].obj, Ebook
    #     )

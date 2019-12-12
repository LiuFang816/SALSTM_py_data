from .pandas_vb_common import *
from pandas.util.decorators import cache_readonly


class DataFrameAttributes(object):
    goal_time = 0.2

    def setup(self):
        self.df = DataFrame(np.random.randn(10, 6))
        self.cur_index = self.df.index

    def time_get_index(self):
        self.foo = self.df.index

    def time_set_index(self):
        self.df.index = self.cur_index


class CacheReadonly(object):
    goal_time = 0.2

    def setup(self):

        class Foo:

            @cache_readonly
            def prop(self):
                return 5
        self.obj = Foo()

    def time_cache_readonly(self):
        self.obj.prop

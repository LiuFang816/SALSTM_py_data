def product(l):
    x = 1
    for e in l:
        x *= e
    return x


class Counter(object):

    def __init__(self, fmt):
        self.fmt = fmt
        self.next_id = 0

    def next(self):
        name = self.fmt % (self.next_id,)
        self.next_id += 1
        return name

    def __iter__(self):
        for i in xrange(self.next_id):
            yield self.fmt % (i,)

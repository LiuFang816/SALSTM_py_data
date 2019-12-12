from .utils import Counter

class Symbol(object):
    required_fields = ()

    def __init__(self, position, **kwargs):
        self.position = position

        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v


class Output(Symbol):
    required_fields = ('shape',)

class Monad(Symbol):
    required_fields = ('rank', 'expr')

class Dyad(Symbol):
    required_fields = ('rank', 'expr')

class Input(Symbol):
    required_fields = ('shape',)

class Literal(Symbol):
    required_fields = ('shape', 'body')

class ApplyMonad(Symbol):
    required_fields = ('shape', 'op', 'rank', 'y')

class ApplyDyad(Symbol):
    required_fields = ('shape', 'op', 'rank', 'x', 'y')

class Constant(Symbol):
    required_fields = ('shape', 'value')

class AccMonad(Symbol):
    required_fields = ('shape', 'acc', 'v', 'operand')

class AccDyad(Symbol):
    required_fields = ('shape', 'acc', 'v', 'operand')


class Vars(object):

    def __init__(self):
        self.counter = Counter("v%d")
        self.data = {}

    def add(self, value):
        name = self.counter.next()
        self.data[name] = value
        return name

    def __getitem__(self, name):
        return self.data[name]

    def __iter__(self):
        return iter(self.counter)


class Symbols(object):

    def __init__(self, default=None):
        self.data = {}
        if default is not None:
            self.data.update(default)

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value):
        assert name not in self.data
        self.data[name] = value

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, name):
        return name in self.data

    def iteritems(self):
        return self.data.iteritems()



class Table(object):

    def __init__(self, default):
        self.vars = Vars()
        self.symbols = Symbols(default)
        self.inputs = Symbols()
        self.outputs = Symbols()
        self.use_graphs = {}
        self.grads = {}

    def get_use_graph(self, v):
        graph = self.use_graphs.get(v, None)
        if graph is None:
            assert self.vars[v].shape == ()
            graph = {}

            visited = set()
            queue = [v]

            while queue:
                u = queue.pop(0)
                if u in visited:
                    continue
                visited.add(u)
                var = self.vars[u]

                if isinstance(var, ApplyMonad):
                    queue.append(var.y)
                    graph[var.y] = graph.get(var.y,())+((u,'y'),)
                elif isinstance(var, ApplyDyad):
                    queue.append(var.x)
                    queue.append(var.y)
                    graph[var.x] = graph.get(var.x,())+((u,'x'),)
                    graph[var.y] = graph.get(var.y,())+((u,'y'),)
                elif isinstance(var, Literal):
                    pass
                elif isinstance(var, Input):
                    pass
                else:
                    raise NotImplementedError

            self.use_graphs[v] = graph

        return graph


    def get_acc(self, acc, v, operand):
        val = self.vars[v]
        shape = self.vars[getattr(val, operand)].shape

        if isinstance(val, ApplyMonad):
            return self.vars.add(
                AccMonad(
                    None,
                    shape = shape,
                    acc = acc,
                    v = v,
                    operand = operand
                )
            )
        elif isinstance(val, ApplyDyad):
            return self.vars.add(
                AccDyad(
                    None,
                    shape = shape,
                    acc = acc,
                    v = v,
                    operand = operand
                )
            )
        else:
            raise NotImplementedError


    def get_grad(self, v, w):
        grad = self.grads.get((v,w), None)
        if grad is not None:
            return grad

        if v == w:
            return self.vars.add(
                Constant(
                    None,
                    shape=(),
                    value=(1.0,)))

        graph = self.get_use_graph(v)
        assert w in graph

        u, operand = graph[w][0]
        result = self.get_acc(self.get_grad(v,u), u, operand)

        for u, operand in graph[w][1:]:
            result2 = self.get_acc(self.get_grad(v,u), u, operand)
            result = self.add_var(
                ApplyDyad(
                    None,
                    shape = self.vars[result].shape,
                    op = self.symbols['+'],
                    rank = (),
                    x = result,
                    y = result2))

        self.grads[(v,w)] = result
        return result

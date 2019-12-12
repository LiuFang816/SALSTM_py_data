class Node(object):
    required_fields = ()

    def __init__(self, position, **kwargs):
        self.position = position

        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v


class Declare(Node):
    required_fields = ('name','value')

class Op(Node):
    required_fields = ('name','rank','value')

class Assign(Node):
    required_fields = ('name','value')

class Integer(Node):
    required_fields = ('n',)

class Number(Node):
    required_fields = ('n',)

class Unknown(Node):
    pass

class Name(Node):
    required_fields = ('s',)

class Numbers(Node):
    required_fields = ('body',)

class Exprs(Node):
    required_fields = ('body',)

class Rank(Node):
    pass

class Arg(Node):
    required_fields = ('s',)

import os.path
import sys
from ply import lex, yacc
from . import ast


class Position(object):
    __slots__ = ('filename', 'lineno', 'col_offset')

    def __init__(self, filename, lineno, col_offset):
        self.filename = filename
        self.lineno = lineno
        self.col_offset = col_offset


class BaseParser(object):
    reserved = ()

    def __init__(self, filename):
        self.filename = filename
        self.tokens = [ r.upper() for r in self.reserved ] + [ a[2:] for a in dir(self) if a[:2] == 't_' and a[2:].isupper() ]
        self.lexer = lex.lex(module=self, debug=False)
        self.parser = yacc.yacc(
            module=self,
            debug=False,
            write_tables=False,
            picklefile=os.path.splitext(
                sys.modules[self.__class__.__module__].__file__
                )[0]+'.parsetab')


    def t_error(self, t):
        raise SyntaxError(
            "Illegal character '%s'!" % t.value[0],
            ( self.filename,
              t.lineno,
              self.col_offset(t.lexpos),
              self.line_of(t)
            ))


    def p_error(self, p):
        raise SyntaxError(
            "Invalid Syntax",
            ( self.filename,
              p.lineno,
              self.col_offset(p.lexpos),
              self.line_of(p)
            ))


    def line_of(self, t):
        last_cr = self.lexer.lexdata.rfind('\n', 0, t.lexpos)
        next_cr = self.lexer.lexdata.find('\n', t.lexpos)
        if next_cr < 0:
            next_cr = None
        return self.lexer.lexdata[last_cr+1: next_cr]


    def col_offset(self, lexpos):
        last_cr = self.lexer.lexdata.rfind('\n', 0, lexpos)
        if last_cr < 0:
            last_cr = 0
        return lexpos - last_cr


    def position(self, p, i):
        return Position(
            filename = self.filename,
            lineno = p.lineno(i),
            col_offset = self.col_offset(p.lexpos(i)))


    def parse(self, data):
        return self.parser.parse(data, lexer=self.lexer, tracking=True)



class Parser(BaseParser):
    reserved = ('ARG',)

    literals = ['(', ')', '"']

    t_INTEGER = r'[0-9]+'
    t_NUMBER = r'[0-9]+[.][0-9]*([eE]-?[0-9]+)?'
    t_ASSIGN = r':='
    t_DECLARE = r'::'
    t_UNKNOWN = r'_'
    t_ignore = ' \t'

    def t_NAME(self, t):
        r'[a-zA-Z+\-*/%<>\.][0-9a-zA-Z+\-*/%<>\.]*'
        t.type = 'ARG' if t.value in ('x','y') else 'NAME'
        return t

    def t_ignore_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def p_file(self, p):
        """file : file stmt"""
        p[0] = p[1] + (p[2],)


    def p_file_empty(self, p):
        """file : """
        p[0] = ()

    def p_stmt_declare(self, p):
        """stmt : NAME DECLARE expr"""
        p[0] = ast.Declare(
            self.position(p,2),
            name=p[1],
            value=p[3])

    def p_stmt_assign_op(self, p):
        """stmt : NAME '"' numbers ASSIGN expr"""
        p[0] = ast.Op(
            self.position(p,4),
            name=p[1],
            rank=ast.Numbers(self.position(p,3), body=p[3]),
            value=p[5])

    def p_stmt_assign(self, p):
        """stmt : NAME ASSIGN expr"""
        p[0] = ast.Assign(
            self.position(p,2),
            name=p[1],
            value=p[3])

    def p_expr_name(self, p):
        """expr : NAME"""
        p[0] = ast.Name(
            self.position(p,1),
            s=p[1])

    def p_expr_arg(self, p):
        """expr : ARG"""
        p[0] = ast.Arg(
            self.position(p,1),
            s=p[1])

    def p_expr_exprs(self, p):
        """expr : '(' exprs ')'"""
        p[0] = ast.Exprs(
            self.position(p,1),
            body=p[2])

    def p_expr_rank(self, p):
        """expr : '"'"""
        p[0] = ast.Rank(self.position(p,1))

    def p_expr_numbers(self, p):
        """expr : numbers"""
        p[0] = ast.Numbers(
            self.position(p,1),
            body=p[1])

    def p_numbers(self, p):
        """numbers : numbers number"""
        p[0] = p[1] + (p[2],)

    def p_numbers_number(self, p):
        """numbers : number"""
        p[0] = (p[1],)

    def p_number(self, p):
        """number : NUMBER"""
        p[0] = ast.Number(
            self.position(p,1),
            n=p[1])

    def p_number_integer(self, p):
        """number : INTEGER"""
        p[0] = ast.Integer(
            self.position(p,1),
            n=p[1])

    def p_number_unknown(self, p):
        """number : UNKNOWN"""
        p[0] = ast.Unknown(self.position(p,1))

    def p_exprs(self, p):
        """exprs : exprs expr"""
        p[0] = p[1] + (p[2],)

    def p_exprs_one(self, p):
        """exprs : expr"""
        p[0] = (p[1],)

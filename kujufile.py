# -*- coding: utf-8 -*-
"""A parser for Microsoft Train Simulator/Open Rails (SIMISA@@@) text files."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from re import match


class ParserException(Exception):
    def __init__(self, subject, message):
        self.subject = subject
        self.message = message
    def __repr__(self):
        if self.subject is None:
            return self.message
        else:
            return f'{self.message}: {self.subject}'
    def __str__(self):
        return self.__repr__()


def load(fp):
    pass


class Token:
    def __str__(self):
        return self.__repr__()

@dataclass
class HeaderToken(Token):
    string: str
    RE = r'SIMISA@@@@@@@@@@JINX0(\w)0t______'
    def __post_init__(self):
        assert HeaderToken.match(self.string)
    def __repr__(self):
        return self.string
    def match(s): return match(HeaderToken.RE, s) is not None

class LParenToken(Token):
    def __repr__(self):
        return '('

class RParenToken(Token):
    def __repr__(self):
        return ')'

class PlusToken(Token):
    def __repr__(self):
        return '+'

@dataclass
class StringToken(Token):
    value: str
    def __repr__(self):
        return self.value.__repr__()

@dataclass
class IntegerToken(Token):
    value: int
    def __repr__(self):
        return self.value.__repr__()

@dataclass
class FloatToken(Token):
    value: float
    def __repr__(self):
        return self.value.__repr__()

def lexer(chars):
    class State(Enum):
        NORMAL = 0
        ID = 10
        QUOTE = 11
        QUOTE_ESCAPE = 12
        NUMBER = 13
        COMMENT_SLASH = 20
        COMMENT = 21
    state = State.NORMAL
    lexeme = None
    def evaluate():
        nonlocal state, lexeme
        if state == State.ID:
            # Cheat the SIMISA@@@ header by treating it as a StringToken.
            if HeaderToken.match(lexeme):
                ret = HeaderToken(lexeme)
            else:
                ret = StringToken(lexeme)
        elif state == State.QUOTE:
            ret = StringToken(lexeme)
        elif state == State.NUMBER:
            if match(r'[a-fA-F\d]{8}$', lexeme):
                ret = IntegerToken(int(lexeme, 16))
            elif match(r'-?\d+$', lexeme):
                ret = IntegerToken(int(lexeme, 10))
            elif match(r'-?(\d*\.\d+|\d+\.\d*)$', lexeme):
                ret = FloatToken(float(lexeme))
            else:
                raise ParserException(lexeme, 'bad number')
        else:
            raise ParserException(None, f'bad lexer state: {state} {lexeme}')
        state = State.NORMAL
        lexeme = None
        return ret

    for ch in chars:
        if state == State.NORMAL:
            if ch == '(':
                yield LParenToken()
            elif ch == ')':
                yield RParenToken()
            elif ch == '+':
                yield PlusToken()
            elif ch == '"':
                state = State.QUOTE
                lexeme = ''
            elif ch.isnumeric() or ch == '.' or ch == '-':
                state = State.NUMBER
                lexeme = ch
            elif ch.isalpha():
                state = state.ID
                lexeme = ch
            elif ch == '/':
                state = state.COMMENT_SLASH
        elif state == State.ID:
            if (ch.isalpha() or ch.isnumeric() or ch == '.' or ch == '_'
                    or ch == '@'):
                lexeme += ch
            elif ch == '(':
                yield evaluate()
                yield LParenToken()
            elif ch == ')':
                yield evaluate()
                yield RParenToken()
            elif ch == '+':
                yield evaluate()
                yield PlusToken()
            elif ch == '"':
                yield evaluate()
                state = State.QUOTE
                lexeme = ''
            elif ch == '/':
                yield evaluate()
                state = state.COMMENT_SLASH
            else:
                yield evaluate()
        elif state == State.QUOTE:
            if ch == '\\':
                state = State.QUOTE_ESCAPE
            elif ch == '"':
                yield evaluate()
            else:
                lexeme += ch
        elif state == State.QUOTE_ESCAPE:
            if ch == 'n':
                lexeme += '\n'
            else:
                lexeme += ch
            state = State.QUOTE
        elif state == State.NUMBER:
            if ch.isnumeric() or ch == '.' or ch == '-':
                lexeme += ch
            elif ch == '(':
                yield evaluate()
                yield LParenToken()
            elif ch == ')':
                yield evaluate()
                yield RParenToken()
            elif ch == '+':
                yield evaluate()
                yield PlusToken()
            elif ch == '"':
                yield evaluate()
                state = State.QUOTE
                lexeme = ''
            elif ch == '/':
                yield evaluate()
                state = state.COMMENT_SLASH
            else:
                yield evaluate()
        elif state == State.COMMENT_SLASH:
            if ch == '/':
                state = State.COMMENT
            else:
                raise ParserException(None, 'unexpected /')
        elif state == State.COMMENT:
            if ch == '\n' or ch == '\r':
                state = State.NORMAL

    if lexeme is not None:
        yield evaluate()


class TreeNode:
    def __str__(self):
        return self.__repr__()

@dataclass
class TreeMap(TreeNode):
    name: str
    dimensions: tuple
    items: list
    def __repr__(self):
        ret = ' '.join([self.name, '(', 'x'.join(str(d) for d in self.dimensions)])
        ret += '\n'
        def indent(text):
            idnt = ' '*8
            return '\n'.join(idnt + l for l in text.splitlines())
        ret += '\n'.join(indent(str(item)) for item in self.items)
        ret += '\n)'
        return ret

@dataclass
class TreeVector(TreeNode):
    name: str
    items: list
    def __repr__(self):
        l = [self.name, '('] + [str(item) for item in self.items] + [')']
        return ' '.join(l)

@dataclass
class TreeScalar(TreeNode):
    value: object
    def __add__(self, other):
        return TreeScalar(self.value + other.value)
    def __repr__(self):
        return self.value.__repr__()

class TreeOp(Enum):
    PLUS = 0

@dataclass
class TreeInfix(TreeNode):
    lchild: TreeNode
    op: TreeOp
    rchild: TreeNode
    def __repr__(self):
        if self.op == TreeOp.PLUS:
            op_s = '+'
        else:
            assert False
        return f'{self.lchild} {op_s} {self.rchild}'

def parser(tokens):
    itokens = iter(tokens)
    first = next(itokens)
    if not isinstance(first, HeaderToken):
        raise ParserException(first, 'first token wasn\'t a SIMISA@@@ header')
    return translate(parse_parens(itokens, ''))

def parse_parens(itokens, name):
    until_paren = []
    children = []
    dimensions = None
    for token in itokens:
        until_paren.append(token)
        if isinstance(token, LParenToken):
            if len(until_paren) < 2:
                raise ParserException(None, 'extraneous (')
            elif dimensions is None:
                bad_token = next(
                    (t for t in until_paren[:-2] if not isinstance(t, IntegerToken)),
                    None)
                if bad_token is not None:
                    raise ParserException(
                        bad_token, 'array dimensions must be integers')

                dimensions = tuple(t.value for t in until_paren[:-2])

            name_t = until_paren[-2]
            if not isinstance(name_t, StringToken):
                raise ParserException(name_t, 'expected a string')

            children.append(parse_parens(itokens, name_t.value))
            until_paren = []
        elif isinstance(token, RParenToken):
            break
    if children == []:
        return TreeVector(name, list(parse_vector(until_paren[:-1])))
    else:
        assert dimensions is not None
        return TreeMap(name, dimensions, children)

def parse_vector(tokens):
    class State(Enum):
        NO_LHAND = 0
        LHAND = 1
        INFIX_PLUS = 10
    state = State.NO_LHAND
    item = None
    for token in tokens:
        if state == State.NO_LHAND:
            if (isinstance(token, StringToken)
                    or isinstance(token, IntegerToken)
                    or isinstance(token, FloatToken)):
                item = TreeScalar(token.value)
                state = State.LHAND
            else:
                raise ParserException(token, 'unexpected token')
        elif state == State.LHAND:
            if isinstance(token, PlusToken):
                state = State.INFIX_PLUS
            elif (isinstance(token, StringToken)
                    or isinstance(token, IntegerToken)
                    or isinstance(token, FloatToken)):
                yield item
                item = TreeScalar(token.value)
            else:
                yield item
                item = None
        elif state == State.INFIX_PLUS:
            if (isinstance(token, StringToken)
                    or isinstance(token, IntegerToken)
                    or isinstance(token, FloatToken)):
                item = TreeInfix(item, TreeOp.PLUS, TreeScalar(token.value))
                state = State.LHAND
            else:
                raise ParserException(token, 'unexpected token')
    if state == State.INFIX_PLUS:
        raise ParserException(None, f'hanging infix operator: {state}')
    if item is not None:
        yield item

def translate(tree):
    if isinstance(tree, TreeScalar):
        return tree.value
    elif isinstance(tree, TreeInfix):
        assert tree.op == TreeOp.PLUS
        return translate(tree.lchild) + translate(tree.rchild)
    elif isinstance(tree, TreeVector):
        if len(tree.items) == 1:
            return translate(tree.items[0])
        else:
            return tuple(translate(item) for item in tree.items)
    elif isinstance(tree, TreeMap):
        n_dim = len(tree.dimensions)
        if n_dim == 0:
            return form_dict(tree)
        elif n_dim == 1:
            return form_1darray(tree)
        elif n_dim == 2:
            return form_2darray(tree)
        else:
            raise ParserException(tree, f'unsupported array {n_dim} > 2')
    else:
        assert False

def translate_array(tree):
    if isinstance(tree, TreeMap):
        n_dim = len(tree.dimensions)
        if n_dim == 0:
            return None, form_dict(tree)
        elif n_dim == 1:
            return tree.dimensions[0], form_dict(tree)
        elif n_dim == 2:
            return None, form_2darray(tree)
        else:
            raise ParserException(tree, f'unsupported array {n_dim} > 2')
    elif isinstance(tree, TreeVector):
        return None, translate(tree)
    else:
        assert False

def form_dict(tree):
    assert isinstance(tree, TreeMap)
    dd = defaultdict(lambda: [])
    for item in tree.items:
        dd[item.name].append(translate(item))
    return {k: v[0] if len(v) == 1 else v for (k, v) in dd.items()}

def form_1darray(tree):
    alen = tree.dimensions[0]
    if alen != len(tree.items):
        raise ParserException(
            tree, f'mismatched array length {len(tree.items)} != {alen}')
    assert tree.items != []

    ldd = defaultdict(lambda: defaultdict(lambda: []))
    for i, item in enumerate(tree.items):
        idx, item_t = translate_array(item)
        ldd[item.name][idx if idx is not None else i].append(item_t)
    def dict_to_list(d): return sum([d[k] for k in sorted(d.keys())], [])
    return {name: dict_to_list(ld) for (name, ld) in ldd.items()}

def form_2darray(tree):
    alenx = tree.dimensions[0] + 1
    aleny = tree.dimensions[1] + 1
    if aleny != len(tree.items):
        raise ParserException(
            tree, f'mismatched array y-length {len(tree.items)} != {alen}')

    assert tree.items != []
    first = tree.items[0]
    assert isinstance(first, TreeVector)
    aname = first.name
    bad_item = next((item for item in tree.items
                     if item.name != aname or len(item.items) != alenx), None)
    if bad_item is not None:
        raise ParserException(bad_item, 'non-uniform array of objects')

    return form_dict(tree)


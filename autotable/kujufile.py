# -*- coding: utf-8 -*-
"""A parser for Microsoft Train Simulator/Open Rails (SIMISA@@@) text files."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from re import match


class ParserException(RuntimeError):
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


class Node:
    def __str__(self):
        return self.__repr__()

@dataclass
class Object(Node):
    """A dict-like container denoted by parantheses in MSTS data files.

    Examples:

    Name ("Acela Express trainset")

    Wagon (
            WagonData ( AcelaEndCar Acela )
            UiD ( 1 )
    )
    """

    name: str
    items: list

    def __repr__(self):
        def indent(text):
            idnt = ' '*8
            return '\n'.join(idnt + l for l in text.splitlines())
        if all(isinstance(item, Scalar) or isinstance(item, Infix)
               for item in self.items):
            return ' '.join([self.name, '(']
                            + [str(item) for item in self.items]
                            + [')'])
        else:
            return (f'{self.name} (\n'
                    + '\n'.join(indent(str(item)) for item in self.items)
                    + '\n)')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        """When indexed by integer, this function will return the i'th (by source
        order) descendant, whether string, number, or Object. When indexed by
        string, it will return a list of all descendant Objects with a matching
        name.

        As a convenience, if the requested name is exclusive to a single Object,
        then this function will return that Object directly instead of wrapping
        it in a list. In addition, if said Object consists of a single item, then
        this function will return that item directly instead of wrapping it in an
        Object.
        """
        if isinstance(key, int):
            return self.items[key]
        elif isinstance(key, str):
            sel = [item for item in self.items
                   if (isinstance(item, Object)
                       and item.name.casefold() == key.casefold())]
            if sel == []:
                raise KeyError
            elif len(sel) == 1:
                item = sel[0]
                if (isinstance(item, Object) and len(item) == 1
                        and not isinstance(item[0], Object)):
                    return Object._evaluate(item[0])
                else:
                    return Object._evaluate(item)
            else:
                return [Object._evaluate(item) for item in sel]

    def __contains__(self, item):
        return any(isinstance(i, Object) and i.name.casefold() == item.casefold()
                   for i in self.items)

    def values(self):
        """Get a list of all descendants that are not other Objects.

        Preserves source order, but provides no information about the positions
        of excluded Objects.
        """
        sel = [item for item in self.items if not isinstance(item, Object)]
        return [Object._evaluate(item) for item in sel]

    def _evaluate(item):
        if isinstance(item, Object):
            return item
        elif isinstance(item, Scalar):
            return item.value
        elif isinstance(item, Infix):
            if item.op == Op.PLUS:
                return Object._evaluate(item.lchild) + Object._evaluate(item.rchild)
            else:
                assert False
        else:
            assert False

@dataclass
class Scalar(Node):
    value: object
    def __add__(self, other):
        return Scalar(self.value + other.value)
    def __repr__(self):
        return self.value.__repr__()

class Op(Enum):
    PLUS = 0

@dataclass
class Infix(Node):
    lchild: Node
    op: Op
    rchild: Node
    def __repr__(self):
        if self.op == Op.PLUS:
            op_s = '+'
        else:
            assert False
        return f'{self.lchild}{op_s}{self.rchild}'


def load(fp):
    """Deserialize the file-like object ``fp``."""
    return _parse(chain.from_iterable(fp))


def loads(s):
    """Deserialize the string ``s``."""
    return _parse(s)


def _parse(s):

    class Token:
        def __str__(self):
            return self.__repr__()

    @dataclass
    class HeaderToken(Token):
        string: str
        RE = r'SIMISA@@@@@@@@@@JINX0(\w)(\d)t______'
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
            LITERAL = 10
            QUOTE = 11
            QUOTE_ESCAPE = 12
            COMMENT_SLASH = 20
            COMMENT = 21
        state = State.NORMAL
        lexeme = None
        def evaluate():
            nonlocal state, lexeme
            if state == State.LITERAL:
                # Cheat the SIMISA@@@ header by treating it as a StringToken.
                if HeaderToken.match(lexeme):
                    ret = HeaderToken(lexeme)
                elif match(r'[a-fA-F\d]{8}$', lexeme):
                    ret = IntegerToken(int(lexeme, 16))
                elif match(r'-?\d+$', lexeme):
                    ret = IntegerToken(int(lexeme, 10))
                elif match(r'-?(\d*\.\d+|\d+\.\d*)$', lexeme):
                    ret = FloatToken(float(lexeme))
                else:
                    ret = StringToken(lexeme)
            elif state == State.QUOTE:
                ret = StringToken(lexeme)
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
                elif ch.isalpha() or ch.isnumeric() or ch == '.' or ch == '-':
                    state = State.LITERAL
                    lexeme = ch
                elif ch == '/':
                    state = State.COMMENT_SLASH
            elif state == State.LITERAL:
                if (ch.isalpha() or ch.isnumeric() or ch == '.' or ch == '_'
                        or ch == '-' or ch == '@' or ch == '+'):
                    lexeme += ch
                elif ch == '(':
                    yield evaluate()
                    yield LParenToken()
                elif ch == ')':
                    yield evaluate()
                    yield RParenToken()
                elif ch == '"':
                    yield evaluate()
                    state = State.QUOTE
                    lexeme = ''
                elif ch == '/':
                    yield evaluate()
                    state = State.COMMENT_SLASH
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

    def parens(itokens):
        class State(Enum):
            NORMAL = 0
            STRING_L = 1
            SCALAR_L = 2
            INFIX_PLUS = 10
        state = State.NORMAL
        last = None
        for token in itokens:
            if state == State.NORMAL:
                if isinstance(token, StringToken):
                    # Don't push strings immediately, they might be names.
                    last = Scalar(token.value)
                    state = State.STRING_L
                elif (isinstance(token, IntegerToken)
                      or isinstance(token, FloatToken)):
                    last = Scalar(token.value)
                    yield last
                    state = State.SCALAR_L
                elif isinstance(token, RParenToken):
                    break
                else:
                    raise ParserException(token, 'unexpected token')
            elif state == State.STRING_L:
                if isinstance(token, LParenToken):
                    yield Object(last.value, list(parens(itokens)))
                    last = None
                    state = State.NORMAL
                elif isinstance(token, StringToken):
                    yield last
                    last = Scalar(token.value)
                elif (isinstance(token, IntegerToken)
                      or isinstance(token, FloatToken)):
                    yield last
                    last = Scalar(token.value)
                    yield last
                    state = State.SCALAR_L
                elif isinstance(token, PlusToken):
                    # Don't advance "last," that's our lefthand operand.
                    state = State.INFIX_PLUS
                elif isinstance(token, RParenToken):
                    yield last
                    break
                else:
                    raise ParserException(token, 'unexpected token')
            elif state == State.SCALAR_L:
                if isinstance(token, StringToken):
                    # Don't push strings immediately, they might be names.
                    last = Scalar(token.value)
                    state = State.STRING_L
                elif (isinstance(token, IntegerToken)
                      or isinstance(token, FloatToken)):
                    last = Scalar(token.value)
                    yield last
                elif isinstance(token, RParenToken):
                    break
                elif isinstance(token, PlusToken):
                    # Don't advance "last," that's our lefthand operand.
                    state = State.INFIX_PLUS
                else:
                    raise ParserException(token, 'unexpected token')
            elif state == State.INFIX_PLUS:
                if isinstance(token, StringToken):
                    last = Infix(last, Op.PLUS, Scalar(token.value))
                    state = State.STRING_L
                elif (isinstance(token, IntegerToken)
                      or isinstance(token, FloatToken)):
                    last = Infix(last, Op.PLUS, Scalar(token.value))
                    yield last
                    state = State.SCALAR_L
                elif isinstance(token, RParenToken):
                    raise ParserException(token, 'hanging +')
                else:
                    raise ParserException(token, 'unexpected token')
            else:
                assert False

    tokens = lexer(s)
    first = next(tokens)
    if not isinstance(first, HeaderToken):
        raise ParserException(first, 'first token wasn\'t a SIMISA@@@ header')
    return Object('', list(parens(tokens)))


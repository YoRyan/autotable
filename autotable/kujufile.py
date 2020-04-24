# -*- coding: utf-8 -*-
"""A parser for Microsoft Train Simulator/Open Rails (SIMISA@@@) text files."""

import re
import typing as typ
from dataclasses import dataclass
from enum import Enum
from itertools import chain


class ParserException(RuntimeError):
    def __init__(self, subject, message):
        self.subject: typ.Any = subject
        self.message: str = message
    def __repr__(self) -> str:
        if self.subject is None:
            return self.message
        else:
            return f'{self.message}: {self.subject}'
    def __str__(self) -> str: return self.__repr__()


class Node:
    def __str__(self) -> str: return self.__repr__()

@dataclass
class Scalar(Node):
    _value: object

    Value = typ.Union[str, int, float]

    def __repr__(self) -> str: return self._value.__repr__()

    def value(self) -> Value:
        if (isinstance(self._value, str)
                or isinstance(self._value, int)
                or isinstance(self._value, float)):
            return self._value
        else:
            raise TypeError(f'unexpected scalar type: {type(self._value)}')

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
    _items: typ.Sequence[Node]

    Evaluated = typ.Union['Object', Scalar.Value]

    def __repr__(self) -> str:
        def indent(text: str) -> str:
            idnt = ' '*8
            return '\n'.join(idnt + l for l in text.splitlines())
        if all(isinstance(item, Scalar) or isinstance(item, Infix)
               for item in self._items):
            return ' '.join([self.name, '(']
                            + [str(item) for item in self._items]
                            + [')'])
        else:
            return (f'{self.name} (\n'
                    + '\n'.join(indent(str(item)) for item in self._items)
                    + '\n)')

    def __len__(self) -> int: return len(self._items)

    def __getitem__(self, key: typ.Union[str, int]) \
            -> typ.Union[Node, Evaluated, typ.Sequence[Evaluated]]:
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
            return self._items[key]
        elif isinstance(key, str):
            sel = list(filter(
                lambda item: (isinstance(item, Object)
                              and item.name.casefold() == str(key).casefold()),
                self._items))
            if sel == []:
                raise KeyError
            elif len(sel) == 1:
                item = sel[0]
                if isinstance(item, Object) and len(item) == 1:
                    single = item._items[0]
                    return Object._evaluate(
                        item if isinstance(single, Object) else single)
                else:
                    return Object._evaluate(item)
            else:
                return [Object._evaluate(item) for item in sel]

    def __contains__(self, item: str) -> bool:
        return any(isinstance(i, Object) and i.name.casefold() == item.casefold()
                   for i in self._items)

    def get(self, key: str, default: typ.Any) -> typ.Any:
        return self[key] if key in self else default

    def values(self) -> typ.Sequence[Evaluated]:
        """Get a list of all descendants that are not other Objects.

        Preserves source order, but provides no information about the positions
        of excluded Objects.
        """
        sel = [item for item in self._items if not isinstance(item, Object)]
        return [Object._evaluate(item) for item in sel]

    def _evaluate(item: Node) -> Evaluated:
        if isinstance(item, Object):
            return item
        elif isinstance(item, Scalar):
            return item.value()
        elif isinstance(item, Infix):
            if item.op == Op.PLUS:
                leval = Object._evaluate(item.lchild)
                reval = Object._evaluate(item.rchild)
                if isinstance(leval, str) and isinstance(reval, str):
                    return leval + reval
                elif isinstance(leval, int) and isinstance(reval, int):
                    return leval + reval
                elif isinstance(leval, float) and isinstance(reval, float):
                    return leval + reval
                else:
                    raise TypeError(f"cannot add '{leval}' to '{reval}'")
            else:
                assert False
        else:
            assert False

class Op(Enum):
    PLUS = 0

@dataclass
class Infix(Node):
    lchild: Node
    op: Op
    rchild: Node
    def __repr__(self) -> str:
        if self.op == Op.PLUS:
            op_s = '+'
        else:
            assert False
        return f'{self.lchild}{op_s}{self.rchild}'


def load(fp: typ.TextIO):
    """Deserialize the file-like object ``fp``."""
    return _parse(chain.from_iterable(fp))


def loads(s: typ.Iterable[str]):
    """Deserialize the string ``s``."""
    return _parse(s)


def _parse(s: typ.Iterable[str]):
    class Token:
        def __str__(self) -> str: return self.__repr__()

    @dataclass
    class HeaderToken(Token):
        string: str
        def __repr__(self) -> str: return self.string

    class LParenToken(Token):
        def __repr__(self) -> str: return '('

    class RParenToken(Token):
        def __repr__(self) -> str: return ')'

    class PlusToken(Token):
        def __repr__(self) -> str: return '+'

    @dataclass
    class StringToken(Token):
        value: str
        def __repr__(self) -> str: return self.value.__repr__()

    @dataclass
    class IntegerToken(Token):
        value: int
        def __repr__(self) -> str: return self.value.__repr__()

    @dataclass
    class FloatToken(Token):
        value: float
        def __repr__(self) -> str: return self.value.__repr__()

    def lexer(chars: typ.Iterable[str]) -> typ.Generator[Token, None, None]:
        class State(Enum):
            NORMAL = 0
            LITERAL = 10
            LITERAL_SLASH = 11
            QUOTE = 20
            QUOTE_ESCAPE = 21
            COMMENT_SLASH = 30
            COMMENT = 31
        state: State = State.NORMAL
        lexeme: typ.Optional[str] = None
        def evaluate() -> Token:
            nonlocal state, lexeme
            assert lexeme is not None
            ret: Token
            if state == State.LITERAL or state == State.LITERAL_SLASH:
                # Cheat the SIMISA@@@ header by treating it as a StringToken.
                if re.match(r'SIMISA@@@@@@@@@@JINX0(\w)(\d)t______', lexeme):
                    ret = HeaderToken(lexeme)
                elif re.match(r'[a-fA-F\d]{8}$', lexeme):
                    ret = IntegerToken(int(lexeme, 16))
                elif re.match(r'-?\d+$', lexeme):
                    ret = IntegerToken(int(lexeme, 10))
                elif re.match(r'-?(\d*\.\d+|\d+\.\d*)$', lexeme):
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
                    assert lexeme is not None
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
                    state = State.LITERAL_SLASH
                else:
                    yield evaluate()
            elif state == State.LITERAL_SLASH:
                if ch == '/':
                    yield evaluate()
                    state = State.COMMENT
                else:
                    lexeme += '/'
                    state = State.LITERAL
            elif state == State.QUOTE:
                if ch == '\\':
                    state = State.QUOTE_ESCAPE
                elif ch == '"':
                    yield evaluate()
                else:
                    assert lexeme is not None
                    lexeme += ch
            elif state == State.QUOTE_ESCAPE:
                assert lexeme is not None
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

    def parens(itokens: typ.Iterable[Token]) -> typ.Generator[Node, None, None]:
        class State(Enum):
            NORMAL = 0
            STRING_L = 1
            SCALAR_L = 2
            INFIX_PLUS = 10
        state: State = State.NORMAL
        last: typ.Optional[Node] = None
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
                assert last is not None
                if isinstance(token, LParenToken):
                    if not isinstance(last, Scalar):
                        raise ParserException(last, 'expected a literal token')
                    yield Object(str(last.value()), list(parens(itokens)))
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
                assert last is not None
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


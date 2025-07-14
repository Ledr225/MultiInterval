import re
import operator
import numpy as np
from scipy.stats import gaussian_kde

class Interval:
    def __init__(self, low, high, low_closed=True, high_closed=True):
        self.low = float(low)
        self.high = float(high)
        self.low_closed = low_closed
        self.high_closed = high_closed

    def contains(self, x):
        return (self.low <= x if self.low_closed else self.low < x) and \
               (x <= self.high if self.high_closed else x < x)

    def sample(self, prec):
        if self.high == self.low:
            return [self.low] if self.low_closed and self.high_closed else []
        step = (self.high - self.low) / (prec + 1)
        points = [self.low + i * step for i in range(1, prec + 1)]
        points = [p for p in points if self.contains(p)]
        return points

    def __repr__(self):
        left = '[' if self.low_closed else '('
        right = ']' if self.high_closed else ')'
        return f"{left}{self.low},{self.high}{right}"

def parse_interval(s):
    s = s.strip()
    # Reverting to original parsing for open/closed intervals
    match = re.match(r'^([\[\(])\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*([\]\)])$', s)
    if not match:
        raise ValueError(f"Invalid interval syntax: {s}")
    low_closed = match.group(1) == '['
    high_closed = match.group(4) == ']'
    low = float(match.group(2))
    high = float(match.group(3))
    # Note: user's parse_interval swaps low/high and open/closed status if low > high.
    # This might be intended for symmetry, let's keep that behavior.
    if low > high:
        low, high = high, low
        low_closed, high_closed = high_closed, low_closed
    return Interval(low, high, low_closed, high_closed)

class MultiInterval:
    def __init__(self, intervals):
        # User's provided code does not sort intervals here.
        # Ensure all elements in the list are indeed Interval objects
        if not all(isinstance(i, Interval) for i in intervals):
            raise TypeError("All elements in MultiInterval must be Interval objects.")
        self.intervals = intervals # User's code does not sort here

    def sample(self, prec):
        points = []
        for interval in self.intervals:
            points.extend(interval.sample(prec))
        return points
    
    def __repr__(self):
        return f"{{{', '.join(str(i) for i in self.intervals)}}}"

def parse_multiinterval(s):
    s = s.strip()
    if not (s.startswith('{') and s.endswith('}')):
        raise ValueError("Multi-interval must be enclosed in { }")
    inner = s[1:-1].strip()
    
    parts = []
    depth = 0 # To track nested brackets for intervals
    buf = ''
    for ch in inner:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        
        if ch == ',' and depth == 0: # Only split by comma if not inside an interval
            parts.append(buf.strip())
            buf = ''
        else:
            buf += ch
    
    if buf: # Add the last part if any
        parts.append(buf.strip())

    if not parts or (len(parts) == 1 and not parts[0]): # Handle empty multi-interval {}
        return MultiInterval([])

    intervals = []
    for part in parts:
        if not part: # Skip empty parts that might result from trailing/leading commas
            continue
        intervals.append(parse_interval(part))

    return MultiInterval(intervals)


token_specification = [
    ('NUMBER',   r'-?\d+(\.\d+)?'),
    ('POW',      r'\^'),
    ('FLOORDIV', r'//'),
    ('MOD',      r'%'),
    ('MUL',      r'\*'),
    ('DIV',      r'/'),
    ('ADD',      r'\+'),
    ('SUB',      r'-'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)',),
    # Reverting regex for MULTIINT and INTERVAL to match user's provided file
    ('MULTIINT', r'\{[^}]*\}'), # More general, relies on parse_multiinterval for detail
    ('INTERVAL', r'[\[\(][^\]\)]*[\]\)]'), # More general, relies on parse_interval for detail
    ('SKIP',     r'[ \t]+'),
]

tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex).match

class Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f'Token({self.type}, {self.value})'

def tokenize(code):
    pos = 0
    tokens = []
    while pos < len(code):
        m = get_token(code, pos)
        if not m:
            raise SyntaxError(f'Invalid character or token at position {pos} in expression: "{code}"')
        typ = m.lastgroup
        val = m.group(typ)
        if typ != 'SKIP':
            tokens.append(Token(typ, val))
        pos = m.end()
    return tokens

precedence = {
    'ADD': 1, 'SUB': 1,
    'MUL': 2, 'DIV': 2, 'MOD': 2, 'FLOORDIV': 2,
    'POW': 3
}
right_associative = {'POW'}
operators = set(precedence.keys())

def shunting_yard(tokens):
    output = []
    stack = []
    for token in tokens:
        if token.type in ('MULTIINT', 'INTERVAL', 'NUMBER'):
            output.append(token)
        elif token.type in operators:
            while (stack and stack[-1].type in operators):
                top = stack[-1]
                if ((token.type not in right_associative and precedence[top.type] >= precedence[token.type]) or
                    (token.type in right_associative and precedence[top.type] > precedence[token.type])):
                    output.append(stack.pop())
                else:
                    break
            stack.append(token)
        elif token.type == 'LPAREN':
            stack.append(token)
        elif token.type == 'RPAREN':
            while stack and stack[-1].type != 'LPAREN':
                output.append(stack.pop())
            if not stack:
                raise SyntaxError("Mismatched parentheses")
            stack.pop()
        else: # Handle unexpected tokens that might slip through
            raise SyntaxError(f"Unexpected token in shunting-yard: {token.type} ({token.value})")

    while stack:
        if stack[-1].type in ('LPAREN', 'RPAREN'):
            raise SyntaxError("Mismatched parentheses")
        output.append(stack.pop())
    return output

ops_map = {
    'ADD': operator.add, 'SUB': operator.sub, 'MUL': operator.mul,
    'DIV': operator.truediv, 'MOD': operator.mod,
    'FLOORDIV': operator.floordiv, 'POW': operator.pow,
}

def eval_rpn(rpn_tokens, prec):
    stack = []
    for token in rpn_tokens:
        if token.type == 'NUMBER':
            stack.append

import re
import operator
import numpy as np
from scipy.stats import gaussian_kde

# --- All your original classes and functions go here ---
# (Interval, parse_interval, MultiInterval, parse_multiinterval, Token, tokenize,
#  shunting_yard, eval_rpn, auto_eval_expression, etc.)

class Interval:
    def __init__(self, low, high, low_closed=True, high_closed=True):
        self.low = float(low)
        self.high = float(high)
        self.low_closed = low_closed
        self.high_closed = high_closed

    def contains(self, x):
        return (self.low <= x if self.low_closed else self.low < x) and \
               (x <= self.high if self.high_closed else x < self.high)

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
    match = re.match(r'^([\[\(])\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*([\]\)])$', s)
    if not match:
        raise ValueError(f"Invalid interval syntax: {s}")
    low_closed = match.group(1) == '['
    high_closed = match.group(4) == ']'
    low = float(match.group(2))
    high = float(match.group(3))
    if low > high:
        low, high = high, low
        low_closed, high_closed = high_closed, low_closed
    return Interval(low, high, low_closed, high_closed)

class MultiInterval:
    def __init__(self, intervals):
        self.intervals = intervals

    def sample(self, prec):
        points = []
        for interval in self.intervals:
            points.extend(interval.sample(prec))
        return points

def parse_multiinterval(s):
    s = s.strip()
    if not (s.startswith('{') and s.endswith('}')):
        raise ValueError("Multi-interval must be enclosed in { }")
    inner = s[1:-1].strip()
    parts = []
    depth = 0
    buf = ''
    for ch in inner:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(buf.strip())
            buf = ''
        else:
            buf += ch
    if buf:
        parts.append(buf.strip())
    intervals = [parse_interval(part) for part in parts]
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
    ('RPAREN',   r'\)'),
    ('MULTIINT', r'\{[^}]*\}'),
    ('INTERVAL', r'[\[\(][^\]\)]*[\]\)]'),
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
            raise SyntaxError(f'Invalid character at position {pos}')
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
            stack.append([float(token.value)])
        elif token.type == 'INTERVAL':
            mi = MultiInterval([parse_interval(token.value)])
            stack.append(mi.sample(prec))
        elif token.type == 'MULTIINT':
            mi = parse_multiinterval(token.value)
            stack.append(mi.sample(prec))
        elif token.type in ops_map:
            if len(stack) < 2:
                raise ValueError("Insufficient operands for operator")
            b = stack.pop()
            a = stack.pop()
            func = ops_map[token.type]
            result = []
            for x in a:
                for y in b:
                    if func in (operator.truediv, operator.floordiv) and y == 0:
                        continue
                    try:
                        r = func(x, y)
                        result.append(r)
                    except Exception:
                        continue
            stack.append(result)
        else:
            raise ValueError(f"Unknown token type: {token.type}")
    if len(stack) != 1:
        raise ValueError("Invalid expression structure")
    return stack[0]

def auto_eval_expression(rpn, minimum_sample, max_prec=1000000):
    prec = 1
    result = []
    while prec <= max_prec:
        result = eval_rpn(rpn, prec)
        if len(result) >= minimum_sample:
            return result, prec
        prec *= 2
    return result, prec

def generate_plot_data(values):
    values = np.array(values)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    
    try:
        kde = gaussian_kde(values, bw_method=0.01)
        x_min, x_max = np.min(values), np.max(values)
        padding = (x_max - x_min) * 0.05
        x_vals = np.linspace(x_min - padding, x_max + padding, 500)
        y_vals = kde(x_vals)
        return x_vals.tolist(), y_vals.tolist()
    except Exception:
        return None, None

# --- Main function to be called from JavaScript ---
def run_calculation(expr, min_sample_str):
    try:
        minimum_sample = int(min_sample_str)
        tokens = tokenize(expr)
        rpn = shunting_yard(tokens)
        result, used_prec = auto_eval_expression(rpn, minimum_sample)

        if not result:
            return {"error": "Calculation resulted in no valid data points."}
        
        status_message = f"Calculation successful. Used precision: {used_prec} (found {len(result)} sample points)."
        x_data, y_data = generate_plot_data(result)

        if x_data is None:
             return {"error": "Could not generate a distribution. Result might be a single constant value."}

        return {
            "status": status_message,
            "plot_data": {
                "x": x_data,
                "y": y_data
            }
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
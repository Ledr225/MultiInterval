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
    match = re.match(r'^\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]$', s)
    if not match:
        raise ValueError(f"Invalid interval syntax. Must use square brackets, e.g., [1,2]: {s}")
    
    low_closed = True
    high_closed = True
    
    low = float(match.group(1))
    high = float(match.group(2))
    
    if low > high:
        low, high = high, low
        
    return Interval(low, high, low_closed, high_closed)

class MultiInterval:
    def __init__(self, intervals):
        if not all(isinstance(i, Interval) for i in intervals):
            raise TypeError("All elements in MultiInterval must be Interval objects.")
        self.intervals = sorted(intervals, key=lambda x: x.low)

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
    current_part = []
    bracket_depth = 0
    for char in inner:
        if char == '[':
            bracket_depth += 1
            current_part.append(char)
        elif char == ']':
            bracket_depth -= 1
            current_part.append(char)
        elif char == ',' and bracket_depth == 0:
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
    
    if current_part:
        parts.append("".join(current_part).strip())

    if not parts or (len(parts) == 1 and not parts[0]):
        return MultiInterval([])

    intervals = []
    for part in parts:
        if not part:
            continue
        intervals.append(parse_interval(part))

    return MultiInterval(intervals)


token_specification = [
    ('NUMBER',    r'-?\d+(\.\d+)?'),
    ('POW',       r'\^'),
    ('FLOORDIV', r'//'),
    ('MOD',       r'%'),
    ('MUL',       r'\*'),
    ('DIV',       r'/'),
    ('ADD',       r'\+'),
    ('SUB',       r'-'),
    ('LPAREN',    r'\('),
    ('RPAREN',    r'\)'),
    ('MULTIINT', r'\{(\s*\[[-\d\.]+,\s*[-\d\.]+\]\s*,?)*\s*\[[-\d\.]+,\s*[-\d\.]+\]\s*\}|\{\}'),
    ('INTERVAL', r'\[[-\d\.]+,\s*[-\d\.]+\]'),
    ('SKIP',      r'[ \t]+'),
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
        else:
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
        if len(np.unique(values)) == 1:
            single_val = values[0]
            delta = max(0.01, abs(single_val) * 0.001)
            x_vals = [single_val - delta, single_val, single_val + delta]
            y_vals = [0, 1 / (2 * delta), 0]
            return x_vals, y_vals
        return None, None

    try:
        # Reverted bw_method to 0.01 as per user's provided code for smoothing
        kde = gaussian_kde(values, bw_method=0.01) # Adjusted this line
        x_min_data, x_max_data = np.min(values), np.max(values)
        
        if x_max_data - x_min_data < 1e-9:
            mean_val = np.mean(values)
            x_min_plot = mean_val - 0.1
            x_max_plot = mean_val + 0.1
        else:
            epsilon = (x_max_data - x_min_data) * 0.01
            x_min_plot = x_min_data - epsilon
            x_max_plot = x_max_data + epsilon

        x_vals = np.linspace(x_min_plot, x_max_plot, 500)
        y_vals = kde(x_vals)

        y_vals[y_vals < 0] = 0

        return x_vals.tolist(), y_vals.tolist()
    except Exception as e:
        print(f"Error generating plot data: {e}")
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
        
        x_data, y_data = generate_plot_data(result)

        if x_data is None:
            return {"error": "Could not generate a distribution. Result might be a single constant value or insufficient data for KDE."}

        return {
            "plot_data": {
                "x": x_data,
                "y": y_data
            }
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

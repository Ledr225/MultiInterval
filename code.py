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
        # For simplicity with the new strict parsing, we'll assume intervals parsed are always closed.
        # If open/closed is still desired, the parsing logic needs to be extended to handle it explicitly
        # with square brackets, e.g., '[1,5)' or '(1,5]', but the current request implies '[]' for all.
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
    # Modified regex to only allow square brackets for intervals
    match = re.match(r'^\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]$', s)
    if not match:
        raise ValueError(f"Invalid interval syntax. Must use square brackets, e.g., [1,2]: {s}")
    
    # Since only square brackets are allowed, they are always closed.
    low_closed = True
    high_closed = True
    
    low = float(match.group(1))
    high = float(match.group(2))
    
    # Ensure low <= high; swap if necessary and maintain closed status.
    if low > high:
        low, high = high, low
        # The closed status remains the same since both are square brackets.
        
    return Interval(low, high, low_closed, high_closed)

class MultiInterval:
    def __init__(self, intervals):
        # Ensure all elements in the list are indeed Interval objects
        if not all(isinstance(i, Interval) for i in intervals):
            raise TypeError("All elements in MultiInterval must be Interval objects.")
        self.intervals = sorted(intervals, key=lambda x: x.low) # Optional: keep intervals sorted

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
    
    if current_part: # Add the last part
        parts.append("".join(current_part).strip())

    if not parts or (len(parts) == 1 and not parts[0]): # Handle empty multi-interval {}
        return MultiInterval([])

    intervals = []
    for part in parts:
        if not part: # Skip empty parts that might result from trailing/leading commas
            continue
        intervals.append(parse_interval(part)) # Use the updated parse_interval

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
    ('MULTIINT', r'\{(\s*\[[-\d\.]+,\s*[-\d\.]+\]\s*,?)*\s*\[[-\d\.]+,\s*[-\d\.]+\]\s*\}|\{\}'), # Allows empty {} and intervals with []
    ('INTERVAL', r'\[[-\d\.]+,\s*[-\d\.]+\]'), # Only allows [num,num]
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
        # Handle single point or no data as before (spike or None)
        if len(np.unique(values)) == 1 and len(values) > 0:
            single_val = values[0]
            delta = max(0.01, abs(single_val) * 0.001) 
            x_vals = [single_val - delta, single_val, single_val + delta]
            y_vals = [0, 1 / (2 * delta), 0]
            return x_vals, y_vals
        return None, None

    try:
        x_min_data, x_max_data = np.min(values), np.max(values)

        if x_max_data - x_min_data < 1e-9:
            mean_val = np.mean(values)
            x_min_plot = mean_val - 0.1
            x_max_plot = mean_val + 0.1
        else:
            epsilon = 1e-9 
            x_min_plot = x_min_data - epsilon
            x_max_plot = x_max_data + epsilon

        # Number of bins for the histogram - still high to define sharp features
        num_bins = 1000 
        
        counts, bin_edges = np.histogram(values, bins=num_bins, range=(x_min_plot, x_max_plot), density=True)

        # Create a much denser set of x_coords for plotting (e.g., 5000 points)
        # This will make the straight line segments in Chart.js appear smoother
        num_plot_points = 5000 # Significantly more points for visual smoothness
        dense_x_coords = np.linspace(x_min_plot, x_max_plot, num_plot_points)

        plot_x = []
        plot_y = []

        # Map each dense_x_coord to its corresponding histogram bin value
        for x_coord in dense_x_coords:
            # Find which bin this x_coord belongs to
            # np.digitize returns the index of the bin to which each value in x belongs.
            # The bin_edges are sorted, so we can use searchsorted.
            # Subtract 1 because digitize returns bin_idx (1-indexed for bins), or 0 for values < first bin.
            # np.clip to handle values exactly equal to x_max_plot, placing them in the last bin
            bin_idx = np.searchsorted(bin_edges, x_coord, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, len(counts) - 1) # Ensure index is within valid range

            plot_x.append(x_coord)
            plot_y.append(counts[bin_idx])

        # Ensure y_vals are non-negative
        plot_y = np.array(plot_y)
        plot_y[plot_y < 0] = 0

        return plot_x, plot_y.tolist()
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
        
        # The 'status' field is included but JS will hide it on success.
        status_message = "Calculation completed." 
        x_data, y_data = generate_plot_data(result)

        if x_data is None:
             return {"error": "Could not generate a distribution. Result might be a single constant value or insufficient data for plot."}

        return {
            "status": status_message, # This status will be passed, but JS will hide it on success
            "plot_data": {
                "x": x_data,
                "y": y_data
            }
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

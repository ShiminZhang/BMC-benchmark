import sympy
from sympy import symbols, sympify, expand, Add, limit, oo
#the idea: we can calculate the leading order term by just comparing every term with every other, and seeing
#which one dominates.
#term1 and term2, limit as x\to\infty of term1/term2. if the limit
def extract_leading_term(expr : str):
    print(f"Extracting leading term for {expr}")
    x = symbols("x")
    expr = expand(expr)
    expr = sympify(expr)
    terms = expr.as_ordered_terms()
    leading_term = terms[0]
    for t in terms[1:]:
        L = limit(leading_term/t,x,oo)
        if L==0:
            leading_term = t
        elif L.is_finite:
            leading_term = (leading_term + t).simplify()
        elif L == oo:
            continue
    return leading_term

# l = extract_leading_term("x**0.39254+x**2+log(x)+sin(x)")
# print(l)
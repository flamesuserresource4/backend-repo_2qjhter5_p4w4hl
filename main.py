import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import math
from sympy import symbols, sympify, diff, integrate, lambdify, sin, cos, exp

app = FastAPI(title="Calculus Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Materials (static content)
# ---------------------------
materials = [
    {
        "slug": "limits",
        "title": "Limits",
        "summary": "Understand how functions behave as inputs approach a value.",
        "content": "The limit of f(x) as x→a describes the value f(x) approaches near a. Use limit laws to combine limits.",
        "examples": [
            {"problem": "lim_{x→2} (x^2 - 4)/(x-2)", "solution": "Factor numerator: (x-2)(x+1)/(x-2) → x+2 → 4"},
            {"problem": "lim_{x→0} sin(x)/x", "solution": "1"}
        ],
    },
    {
        "slug": "continuity",
        "title": "Continuity",
        "summary": "When small input changes cause small output changes.",
        "content": "A function is continuous at a if lim_{x→a} f(x) = f(a).",
        "examples": [
            {"problem": "Is |x| continuous at 0?", "solution": "Yes, both one-sided limits equal 0 which equals f(0)."}
        ],
    },
    {
        "slug": "derivatives",
        "title": "Derivatives",
        "summary": "Instantaneous rate of change; slope of the tangent line.",
        "content": "The derivative f'(x) measures how f changes with x. Use rules: power, product, quotient, chain.",
        "examples": [
            {"problem": "d/dx x^3", "solution": "3x^2"},
            {"problem": "d/dx sin(x^2)", "solution": "cos(x^2) * 2x"}
        ],
    },
    {
        "slug": "applications-derivatives",
        "title": "Applications of Derivatives",
        "summary": "Optimization, related rates, linearization.",
        "content": "Critical points occur where f'(x)=0 or undefined. Use first/second derivative tests to classify extrema.",
        "examples": [
            {"problem": "Find extrema of x^3-3x", "solution": "f'(x)=3x^2-3=0 → x=±1; classify with f''(x)=6x."}
        ],
    },
    {
        "slug": "integrals",
        "title": "Integrals",
        "summary": "Area accumulation and antiderivatives.",
        "content": "The definite integral accumulates signed area; Fundamental Theorem connects integrals and derivatives.",
        "examples": [
            {"problem": "∫_0^1 2x dx", "solution": "[x^2]_0^1 = 1"}
        ],
    },
    {
        "slug": "techniques-integration",
        "title": "Techniques of Integration",
        "summary": "Substitution, parts, partial fractions.",
        "content": "Choose techniques to simplify integrals into known forms.",
        "examples": [
            {"problem": "∫ x e^{x} dx", "solution": "Integration by parts: (x-1)e^x + C"}
        ],
    },
    {
        "slug": "series",
        "title": "Sequences and Series",
        "summary": "Infinite sums and convergence tests.",
        "content": "Use ratio/root test for convergence of power series and define radius of convergence.",
        "examples": [
            {"problem": "∑ x^n/n!", "solution": "e^x"}
        ],
    },
    {
        "slug": "multivariable",
        "title": "Multivariable Calculus",
        "summary": "Functions of several variables and gradients.",
        "content": "Partial derivatives measure change with respect to one variable holding others constant.",
        "examples": [
            {"problem": "∂/∂x (x^2y)", "solution": "2xy"}
        ],
    },
    {
        "slug": "vectors",
        "title": "Vector Calculus",
        "summary": "Divergence, curl, line and surface integrals.",
        "content": "Green, Stokes, and Divergence theorems relate integrals over domains and boundaries.",
        "examples": [
            {"problem": "∇·F for F=⟨x,y,z⟩", "solution": "3"}
        ],
    },
]

# ---------------------------
# Models
# ---------------------------
class CalcRequest(BaseModel):
    expression: str = Field(..., description="Function of x, e.g., 'sin(x) + x^2'")
    operation: Literal["derivative", "integral"]
    order: int = 1
    a: Optional[float] = Field(None, description="Lower bound for definite integral")
    b: Optional[float] = Field(None, description="Upper bound for definite integral")

class CalcResponse(BaseModel):
    input: str
    operation: str
    result_expression: Optional[str] = None
    definite_value: Optional[float] = None

class PlotRequest(BaseModel):
    expression: str
    x_min: float = -10
    x_max: float = 10
    points: int = 200

class PlotResponse(BaseModel):
    x: List[float]
    y: List[Optional[float]]

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Calculus Web API running"}

@app.get("/api/materials")
def get_materials():
    return materials

@app.get("/api/materials/{slug}")
def get_material(slug: str):
    for m in materials:
        if m["slug"] == slug:
            return m
    raise HTTPException(status_code=404, detail="Topic not found")

@app.post("/api/calc", response_model=CalcResponse)
def calculate(req: CalcRequest):
    x = symbols('x')
    try:
        expr = sympify(req.expression)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid expression: {e}")

    if req.operation == "derivative":
        try:
            res = diff(expr, x, req.order)
            return CalcResponse(input=req.expression, operation="derivative", result_expression=str(res))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Derivative error: {e}")
    elif req.operation == "integral":
        # definite if a and b provided
        try:
            if req.a is not None and req.b is not None:
                val = float(integrate(expr, (x, req.a, req.b)))
                return CalcResponse(input=req.expression, operation="definite_integral", definite_value=val)
            else:
                antideriv = integrate(expr, x)
                return CalcResponse(input=req.expression, operation="indefinite_integral", result_expression=str(antideriv))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Integral error: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported operation")

@app.post("/api/plot", response_model=PlotResponse)
def plot_points(req: PlotRequest):
    x = symbols('x')
    try:
        expr = sympify(req.expression)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid expression: {e}")

    # Use math backend for speed, ensure safe functions available
    f = lambdify(x, expr, modules={"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "abs": abs, "pow": pow})

    pts = max(10, min(2000, req.points))
    step = (req.x_max - req.x_min) / (pts - 1)
    xs: List[float] = []
    ys: List[Optional[float]] = []

    for i in range(pts):
        xi = req.x_min + i * step
        xs.append(xi)
        try:
            yi = float(f(xi))
            # filter NaN/inf
            if math.isfinite(yi):
                ys.append(yi)
            else:
                ys.append(None)
        except Exception:
            ys.append(None)

    return PlotResponse(x=xs, y=ys)

@app.get("/api/presets")
def presets():
    return [
        {"name": "Sine Wave", "expression": "sin(x)", "range": [-2*math.pi, 2*math.pi]},
        {"name": "Parabola", "expression": "x**2", "range": [-10, 10]},
        {"name": "Exponential", "expression": "exp(x)", "range": [-2, 3]},
    ]

@app.get("/test")
def test_database():
    """Test endpoint to check if backend is running (database optional)"""
    return {
        "backend": "✅ Running",
        "database": "ℹ️ Not used for this app",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

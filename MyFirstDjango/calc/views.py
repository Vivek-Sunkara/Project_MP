from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, sympify
import io
import base64
import re

history = []

def simplex_method(c, A, b):
    """
    Solves the Linear Programming problem using the Simplex Method.
    Refer to the detailed code provided earlier.
    """
    num_constraints, num_variables = A.shape

    # Add slack variables to convert inequalities to equalities
    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))

    # Add the objective function row to the tableau
    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    num_total_vars = num_variables + num_constraints

    # Start Simplex iterations
    while True:
        if all(tableau[-1, :-1] >= 0):
            break

        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        if np.all(ratios == np.inf):
            return "The problem is unbounded.", None

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value


def simplex_view(request):
    global history
    result = None
    error = None
    if request.method == 'POST':
        try:
            # Parse inputs from POST request
            c = np.array(list(map(float, request.POST.get('c').split(','))))
            A = np.array([list(map(float, row.split(','))) for row in request.POST.get('A').split(';')])
            b = np.array(list(map(float, request.POST.get('b').split(','))))
            
            # Solve Simplex
            solution, optimal_value = simplex_method(c, A, b)
            if solution is None:
                error = optimal_value
            else:
                result = {'solution': solution.tolist(), 'optimal_value': optimal_value}
                # Save history
                history.append({
                    'input': {'c': c.tolist(), 'A': A.tolist(), 'b': b.tolist()},
                    'output': result
                })
        except Exception as e:
            error = str(e)
    
    # Render with history
    return render(request, 'simplex_form.html', {'result': result, 'error': error, 'history': history})
# Function to parse the objective function from a string
def parse_objective_function(obj_func_str):
    x, y = symbols('x y')
    # Replace 'x' and 'y' with proper syntax for sympy to handle
    obj_func_str = obj_func_str.replace('x', '*x').replace('y', '*y').replace(" ", "")
    
    # Now sympify the modified string
    try:
        obj_expr = sympify(obj_func_str)
    except Exception as e:
        raise ValueError(f"Error parsing objective function: {str(e)}")
    
    return obj_expr, x, y
history=[]
# Function to parse constraints (this is a simple parser, you may need to refine it)
def parse_constraints(constraints_str):
    constraints = []
    try:
        # Split input by lines
        for line in constraints_str.splitlines():
            # Remove leading and trailing whitespace from each line
            line = line.strip()
            
            # Use regular expression to match the pattern for constraints like '2x + 4y <= 80'
            match = re.match(r'([-\d\w\s\+\*\/]+)(<=|>=|<|>)([-\d\w\s\+\*\/]+)', line)
            if match:
                lhs = match.group(1).strip()
                operator = match.group(2)
                rhs = match.group(3).strip()

                # Ensure rhs is a numeric value
                try:
                    rhs = sympify(rhs)
                except:
                    raise ValueError(f"Error parsing right-hand side value of constraint: {rhs}")

                constraints.append((lhs, rhs, operator))
            else:
                raise ValueError(f"Invalid constraint format: {line}")
        return constraints
    except Exception as e:
        raise ValueError(f"Error parsing constraints: {e}")

# Function to find intersection points of constraints (simplified)
def find_intersection_points(parsed_constraints):
    x, y = symbols('x y')
    points = []
    try:
        for i, (lhs1, rhs1, op1) in enumerate(parsed_constraints):
            for j, (lhs2, rhs2, op2) in enumerate(parsed_constraints):
                if i >= j:
                    continue  # Skip duplicate pairs
                # Convert strings like '2x + 4y' into sympy expressions
                lhs1_expr = sympify(lhs1.replace('x', '*x').replace('y', '*y'))
                lhs2_expr = sympify(lhs2.replace('x', '*x').replace('y', '*y'))
                eq1 = Eq(lhs1_expr, rhs1)
                eq2 = Eq(lhs2_expr, rhs2)
                solution = solve((eq1, eq2), (x, y))
                if solution:
                    points.append((float(solution[x]), float(solution[y])))
    except Exception as e:
        raise ValueError(f"Error finding intersection points: {e}")
    
    return points

# Function to check if a point satisfies the constraints
def is_feasible(point, parsed_constraints):
    x, y = point
    for lhs, rhs, operator in parsed_constraints:
        expr = sympify(lhs.replace('x', '*x').replace('y', '*y')).subs({'x': x, 'y': y})
        if operator == '<=' and expr > rhs:
            return False
        elif operator == '>=' and expr < rhs:
            return False
        elif operator == '<' and expr >= rhs:
            return False
        elif operator == '>' and expr <= rhs:
            return False
    return True

# Function to plot the solution graph
def plot_solution(obj_func, parsed_constraints, optimal_point, optimal_value, opt):
    x_vals = np.linspace(0, 50, 500)
    plt.figure(figsize=(10, 6))
    
    for lhs, rhs, operator in parsed_constraints:
        expr = sympify(lhs.replace('x', '*x').replace('y', '*y'))
        y_vals = [solve(Eq(expr.subs(symbols('x'), val), rhs), symbols('y'))[0] if solve(Eq(expr.subs(symbols('x'), val), rhs), symbols('y')) else None for val in x_vals]
        y_vals = np.array([float(yv) if yv is not None else np.nan for yv in y_vals])
        plt.plot(x_vals, y_vals, label=f"{lhs} {operator} {rhs}")

    # Highlight optimal solution
    plt.scatter(*optimal_point, color='green', s=100, zorder=5, label=f"Optimal Point: {optimal_point}, Z = {optimal_value:.2f}")

    # Plot settings
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.title(f"Graphical Solution ({opt.capitalize()}imization)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Save plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    graph = base64.b64encode(image_png).decode("utf-8")
    return graph

# Helper function to provide a step-by-step breakdown
def generate_step_by_step_solution(obj_func, parsed_constraints, intersection_points, feasible_points, optimal_point, optimal_value, opt):
    steps = []
    
    steps.append("Step 1: Parsing the objective function and constraints.")
    steps.append(f"Objective Function: {obj_func}")
    steps.append(f"Constraints: {parsed_constraints}")
    
    steps.append("Step 2: Finding intersection points of the constraints.")
    steps.append(f"Intersection Points: {intersection_points}")
    
    steps.append("Step 3: Checking the feasibility of the intersection points.")
    steps.append(f"Feasible Points: {feasible_points}")
    
    steps.append("Step 4: Evaluating the objective function at the feasible points.")
    steps.append(f"Optimal Point: {optimal_point} with Objective Value: {optimal_value}")
    
    steps.append(f"Step 5: Plotting the solution graph with the optimal point highlighted.")
    
    return steps
def linear_programming_solver(request):
    global history  # Ensure history is accessible as a global variable

    if request.method == "POST":
        obj_func = request.POST.get("objective_function")
        constraints = request.POST.get("constraints")
        opt = request.POST.get("optimization").lower()

        try:
            obj_expr, x, y = parse_objective_function(obj_func)
            parsed_constraints = parse_constraints(constraints)
            intersection_points = find_intersection_points(parsed_constraints)

            feasible_points = []
            for point in intersection_points:
                if is_feasible(point, parsed_constraints):
                    feasible_points.append(point)

            if not feasible_points:
                raise ValueError("No feasible points found. Check constraints or objective function.")

            objective_values = []
            for point in feasible_points:
                value = obj_expr.subs({'x': point[0], 'y': point[1]})
                objective_values.append(value)

            if opt == "max":
                optimal_value = max(objective_values)
            elif opt == "min":
                optimal_value = min(objective_values)
            else:
                raise ValueError("Invalid optimization type. Choose 'max' or 'min'.")

            optimal_point = feasible_points[objective_values.index(optimal_value)]
            graph = plot_solution(obj_func, parsed_constraints, optimal_point, optimal_value, opt)

            # Generate step-by-step explanation
            steps = generate_step_by_step_solution(obj_func, parsed_constraints, intersection_points, feasible_points, optimal_point, optimal_value, opt)

            # Save input/output to history
            history.append({
                'objective_function': obj_func,
                'constraints': constraints,
                'optimization': opt,
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'steps': steps,
                'graph': graph
            })

            return render(request, 'solver.html', {
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph': graph,
                'steps': steps,
                'obj_func': obj_func,
                'constraints': constraints,
                'opt': opt,
                'history': history,  # Include history in the context
            })

        except Exception as e:
            return render(request, 'solver.html', {'error': str(e), 'history': history})  # Include history in case of error

    return render(request, 'solver.html', {'history': history}) 
# --- Transportation Problem Solver ---
from scipy.optimize import linprog

# Global history for both solvers
history = []

def solve_transportation_problem(cost_matrix, supply, demand):
    cost_matrix = np.array(cost_matrix)
    supply = np.array(supply)
    demand = np.array(demand)

    m, n = cost_matrix.shape
    c = cost_matrix.flatten()

    A_eq = []
    b_eq = []

    for i in range(m):
        row_constraint = [0] * (m * n)
        for j in range(n):
            row_constraint[i * n + j] = 1
        A_eq.append(row_constraint)
        b_eq.append(supply[i])

    for j in range(n):
        col_constraint = [0] * (m * n)
        for i in range(m):
            col_constraint[i * n + j] = 1
        A_eq.append(col_constraint)
        b_eq.append(demand[j])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if result.success:
        solution_matrix = result.x.reshape(m, n)
        return {
            "solution": solution_matrix,
            "total_cost": result.fun,
            "status": result.message,
        }
    else:
        return {
            "solution": None,
            "total_cost": None,
            "status": result.message,
        }

def transportation_view(request):
    global history
    result = None
    error = None
    if request.method == 'POST':
        form_data = request.POST
        try:
            # Parse user inputs
            cost_matrix = [list(map(int, row.split(','))) for row in form_data['cost_matrix'].splitlines()]
            supply = list(map(int, form_data['supply'].split(',')))
            demand = list(map(int, form_data['demand'].split(',')))

            # Solve transportation problem
            result = solve_transportation_problem(cost_matrix, supply, demand)

            # Store history of inputs and results
            history.append({
                'input': {'cost_matrix': cost_matrix, 'supply': supply, 'demand': demand},
                'output': result
            })

        except Exception as e:
            error = str(e)
    
    return render(request, 'form.html', {
        'result': result,
        'error': error,
        'history': history
    })

def home(request):
    return render(request, 'home.html') 
from django.shortcuts import render

# Create your views here.
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sympy import symbols, Eq, solve
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re

@csrf_exempt
def solve_lp_graphical(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            objective = data.get("objective", "").replace(" ", "")  # Remove spaces
            constraints = data.get("constraints", [])
            optimization_type = data.get("optimization_type", "max").lower()  # max or min

            # Input validation
            if not objective:
                return JsonResponse({"success": False, "message": "Objective function is required."})
            if not constraints:
                return JsonResponse({"success": False, "message": "At least one constraint is required."})
            if optimization_type not in ["max", "min"]:
                return JsonResponse({"success": False, "message": "Optimization type must be 'max' or 'min'."})

            # Parse the objective function
            objective_pattern = re.compile(r'([-+]?\d+\.?\d*)([a-zA-Z])')
            obj_coeffs = {match[1]: float(match[0]) for match in objective_pattern.findall(objective)}
            c_x = obj_coeffs.get('x', 0)
            c_y = obj_coeffs.get('y', 0)

            # Parse constraints
            constraint_lines = []
            for constraint in constraints:
                parts = re.split(r'(<=|>=|=)', constraint)
                if len(parts) != 3:
                    raise ValueError(f"Invalid constraint format: {constraint}")

                left = parts[0].strip()
                operator = parts[1]
                right = parts[2].strip()

                try:
                    right_value = float(right)
                except ValueError:
                    raise ValueError(f"Invalid right-hand side in constraint: {right}")

                coeffs = {match[1]: float(match[0]) for match in objective_pattern.findall(left)}
                constraint_lines.append((coeffs.get('x', 0), coeffs.get('y', 0), operator, right_value))

            # Calculate intersection points
            x, y = symbols('x y')
            intersections = []
            for (a1, b1, _, c1), (a2, b2, _, c2) in combinations(constraint_lines, 2):
                eq1 = Eq(a1 * x + b1 * y, c1)
                eq2 = Eq(a2 * x + b2 * y, c2)
                sol = solve((eq1, eq2), (x, y))
                if sol:
                    intersections.append((sol[x], sol[y]))

            # Filter feasible points
            feasible_points = []
            for px, py in intersections:
                if all(
                    (a * px + b * py <= rhs if op == "<=" else a * px + b * py >= rhs)
                    for a, b, op, rhs in constraint_lines
                ):
                    feasible_points.append((px, py))

            # Evaluate the objective function at feasible points
            solutions = []
            for px, py in feasible_points:
                obj_value = c_x * px + c_y * py
                solutions.append((px, py, obj_value))

            # Find the optimal solution
            if optimization_type == "max":
                optimal_solution = max(solutions, key=lambda s: s[2])
            else:
                optimal_solution = min(solutions, key=lambda s: s[2])

            # Plot the constraints and feasible region
            fig, ax = plt.subplots()
            x_vals = np.linspace(0, max([rhs for _, _, _, rhs in constraint_lines]) * 2, 500)

            for a, b, op, rhs in constraint_lines:
                if b != 0:
                    y_vals = (rhs - a * x_vals) / b
                    ax.plot(x_vals, y_vals, label=f"{a}x + {b}y {op} {rhs}")
                else:
                    ax.axvline(x=rhs / a, color='gray', linestyle='--', label=f"x = {rhs / a}")

            ax.scatter(*zip(*feasible_points), color='green', label="Feasible Points")
            ax.scatter(optimal_solution[0], optimal_solution[1], color='red', label="Optimal Solution", zorder=5)
            ax.set_xlim(0, max([rhs for _, _, _, rhs in constraint_lines]) * 1.5)
            ax.set_ylim(0, max([rhs for _, _, _, rhs in constraint_lines]) * 1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Graphical Solution of Linear Programming")
            plt.legend()
            plt.grid()

            # Encode graph as a Base64 image
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            graph_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()

            return JsonResponse({
                "success": True,
                "solution": {
                    "x": optimal_solution[0],
                    "y": optimal_solution[1],
                    "objective": optimal_solution[2]
                },
                "graph": graph_base64
            })

        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)})
    return JsonResponse({"success": False, "message": "Invalid request method."})

def graphical_method(request):
    return render(request, 'Graphical_method_of_solution_to_LP.html')

def home(request):
    return render(request, 'home.html', {'name': 'Raju'})
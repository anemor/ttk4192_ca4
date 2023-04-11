import pyplanning as pp

#domain_file = "/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/domain.pddl"
#problem_file = "/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/problem.pddl"

domain_file = "./domain.pddl"
problem_file = "./problem.pddl"

domain, problem = pp.load_pddl(domain_file, problem_file)
plan = pp.solvers.graph_plan(problem, 1000, True)

if plan is not None:
    print("Plan found:")
    print(plan, "\n")
else:
    print("Planning failed.")
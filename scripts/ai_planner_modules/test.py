import pyplanning as pp

domain_file = "/home/marie/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/domain.pddl"
problem_file = "/home/marie/catkin_ws/src/ca4_ttk4192/scripts/ai_planner_modules/PDDL_domain/problem.pddl"

domain, problem = pp.load_pddl(domain_file, problem_file)
plan = pp.solvers.graph_plan(problem, 1000, False)

if plan is not None:
    print("Plan found:")
    print(plan, "\n")
    #x = plan.split(",")
    #print(x)

    # Reading plan:
    for i in range(len(plan)):
        lst = list(plan[i+1])[0]       # charge-robot(WP0, charger0, turtlebot0, battery0)
        action = lst.action.name
        obj = []
        for o in lst.objects:
            obj.append(str(o))
        
        print(action)       # charge-robot, move, 
        print(obj[0])

else:
    print("Planning failed.")
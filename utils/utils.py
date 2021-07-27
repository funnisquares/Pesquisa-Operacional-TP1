def print_solution(sol):
    if sol[0] == "otima":
        print(sol[0])
        print(f"{round(sol[1], 7):.7f}")

        out = ""
        for s in sol[2]:
            out += f"{round(s, 7):.7f} "
        print(out)
        out = ""
        for s in sol[3]:
            out += f"{round(s, 7):.7f} "
        print(out)
    elif sol[0] == "inviavel":
        print(sol[0])
        out = ""
        for s in sol[1]:
            out += f"{round(s, 7):.7f} "
        print(out)
    else:
        print(sol[0])
        out = ""
        for s in sol[1]:
            out += f"{round(s, 7):.7f} "
        print(out)

        out = ""
        for s in sol[2]:
            out += f"{round(s, 7):.7f} "
        print(out)

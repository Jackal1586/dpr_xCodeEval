import plotly.express as px

x = []
with open("dpr_eval_output.txt") as rp:
    x = [float(i.strip()) / 145275 for i in rp.read().split()]


px.line(x).show()
# input()

"""
----------------------------
Pseudocode for Graphplan:
----------------------------

Graphplan(A, s_i, g)

# Initialization
i       = 0             # iterative variable
noGoods = []            # nabla
P[0]    = s_i           # proposition layer
G       = (P[0],{})     # graph

while ( (g not in P[i]) or (g² union mu*P[i] != {}) ) and (not fixedPoint(G)):
    i += 1
    expand(G)

if (g not in P[i]) or (g² union mu*P[i] != {}):
    return failure

n       = fixedPoint(G)
delK    = 0
Pi      = extract(G, g, i)

while (Pi = failure):
    i += 1
    expand(G)
    Pi = extract(G, g, i)
    if (Pi = failure) and fixedPoint(G):
        if n == delK:
            return failure
        n = delK

return Pi
"""


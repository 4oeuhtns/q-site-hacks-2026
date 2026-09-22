import networkx as nx

def used_logical_qubits(program):
    """returns list of used logical notebooks in order to iterate through"""
    qubits = set()
    for g in program:
        for q in g[1:]:
            qubits.add(q)

    return sorted(qubits)

def depth(routed_program):
    """returns depth"""
    counts = dict.fromkeys(used_logical_qubits(routed_program), 0)

    for g in routed_program:
        if g[0] != '1Q':
            max_val = max(counts[g[1]], counts[g[2]])
            counts[g[1]] = max_val + 1
            counts[g[2]] = max_val + 1

    return max(counts.values()) if counts else 0

def cost(routed_program):
    """returns cost of routed program"""
    swaps = 0
    for g in routed_program:
        if g[0] == 'SWAP':
            swaps += 1
    return swaps + 0.5*depth(routed_program)

def validate(program, graph, placement, routed_program):
    """checks if placed routed program on hardware is equal to original"""
    # placement must cover exactly the logical qubits the program uses
    if set(placement.keys()) != set(used_logical_qubits(program)):
        return False
    # no two logical qubits mapped to physical qubit
    if len(set(placement.values())) != len(placement):
        return False
    # every phys has to actually exist on the hardware
    if not set(placement.values()).issubset(graph.nodes):
        return False

    # how physical qubits map to logical qubits
    phys_to_log = {v:k for k,v in placement.items()}
    decompiled = []
    for g in routed_program:
        if g[0] == 'SWAP':
            if not graph.has_edge(g[1], g[2]):
                return False
            left = phys_to_log.pop(g[1], None)
            right = phys_to_log.pop(g[2], None)
            if right is not None:
                phys_to_log[g[1]] = right
            if left is not None:
                phys_to_log[g[2]] = left
        elif g[0] == '2Q':
            if not graph.has_edge(g[1], g[2]):
                return False
            if g[1] not in phys_to_log or g[2] not in phys_to_log:
                return False
            decompiled.append(('2Q', phys_to_log[g[1]], phys_to_log[g[2]]))
        elif g[0] == '1Q':
            if g[1] not in phys_to_log:
                return False
            decompiled.append(('1Q', phys_to_log[g[1]]))
        else:
            return False

    return decompiled == program

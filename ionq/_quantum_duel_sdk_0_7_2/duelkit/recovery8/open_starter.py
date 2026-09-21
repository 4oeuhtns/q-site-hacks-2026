"""Tiny count-only OPEN-final starter, not a general architecture learner.

The only fitted hypotheses are Rz(q0), RZZ(q0,q1), and identity. This is a
working protocol example for the public local/zz smoke cases. Arbitrary final
attacks need a materially better defender. No hidden data or private score is used.
"""
from __future__ import annotations
import math
from duelkit.quantum import G, experiment_index, validate
from qduel_sdk.rules import OPEN8_RULESET


def run_defender(client, rules):
    if rules.version != OPEN8_RULESET:
        raise ValueError('This teaching starter targets the open eight-qubit profile')
    totals = [[0, 0] for _ in range(4)]  # measured q0 minus/plus outcomes
    mask = 1 << (rules.qubits - 1)
    for stage in range(1, rules.checkpoints + 1):
        available = int(client.status()['available_now'])
        per_setting = available // 4
        requests = []
        for i, (partner, axis) in enumerate((('0', 'X'), ('0', 'Y'), ('1', 'X'), ('1', 'Y'))):
            shots = per_setting + (1 if i < available % 4 else 0)
            if not shots:
                continue
            prep = ['+', partner] + ['0'] * (rules.qubits - 2)
            requests.append(dict(request_id=f'open-example-{stage}-{i}',
                experiment_index=experiment_index(prep, axis + 'Z' * (rules.qubits - 1), rules.qubits),
                shots=shots))
        # This deterministic policy needs no results between the four experiments.
        for i, receipt in enumerate(client.query_batch(requests)):
            counts = receipt['counts']
            minus = sum(n for bits, n in counts.items() if int(bits, 2) & mask)
            totals[i][0] += minus
            totals[i][1] += int(receipt['shots']) - minus
        means = [(plus-minus)/(plus+minus) if plus+minus else 0.0 for minus, plus in totals]
        x0, y0, x1, y1 = means
        local_angle = math.atan2(y0+y1, x0+x1)
        pair_angle = math.atan2(y0-y1, x0+x1)
        candidates = [
            ((), [1., 0., 1., 0.]),
            ((G('rz', (0,), -local_angle),),
             [math.cos(local_angle), math.sin(local_angle)] * 2),
            ((G('rzz', (0,1), -pair_angle),),
             [math.cos(pair_angle), math.sin(pair_angle), math.cos(pair_angle), -math.sin(pair_angle)]),
        ]
        # Observed binomial likelihood only. This is not the private channel score.
        def loss(candidate):
            return -sum(plus*math.log(max(1e-12,(1+v)/2)) + minus*math.log(max(1e-12,(1-v)/2))
                        for (minus,plus),v in zip(totals,candidate[1]))
        patch = min(candidates, key=loss)[0]
        validate(patch, **rules.validation_kwargs())
        client.submit_patch(patch, note='Teaching baseline: only local phase / single ZZ hypotheses')
        client.close_checkpoint()
    client.finish()

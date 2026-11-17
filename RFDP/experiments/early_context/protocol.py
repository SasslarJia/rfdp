"""One frozen stage-switch candidate; no placebo or parameter search."""
from experiments.context_graph import protocol as context

SCHEMA = 'early-context-switch-v1'
ROUTE = 'early_context_switch'
WINDOW = 5
ORDER = ('duke-osnet', 'market-osnet', 'market-transreid', 'duke-transreid')


def recipe(input_id):
    if input_id not in ORDER:
        raise ValueError('Unknown official input')
    cfg = context.recipe('context_graph_early5', input_id)
    cfg.update(adaptation_schema=SCHEMA, adaptation_route=ROUTE,
               runner='experiments.early_context.full_run',
               phase='main' if input_id == ORDER[0] else 'transfer',
               requires_decision=input_id != ORDER[0],
               graph_schedule='context_first5_then_original',
               context_window_new_nodes=WINDOW,
               numerical_solver='phase_certified',
               early_numerical_solver='stable_lu_certified',
               late_numerical_solver='warm_certified',
               requires_rank1_screen=True)
    return cfg


def validate(cfg, require_bound=False):
    expected = recipe(cfg.get('input_id'))
    changed = ('adaptation_schema', 'adaptation_route', 'runner', 'phase',
               'requires_decision', 'graph_schedule', 'context_window_new_nodes',
               'numerical_solver', 'early_numerical_solver', 'late_numerical_solver',
               'requires_rank1_screen')
    for key in changed:
        if cfg.get(key) != expected[key] or type(cfg.get(key)) is not type(expected[key]):
            raise ValueError('Frozen early-context setting differs: ' + key)
    # Reuse the existing immutable two-graph input binding, not its old run plan.
    parent = dict(cfg)
    parent_recipe = context.recipe('context_graph_early5', cfg['input_id'])
    for key in ('adaptation_schema', 'adaptation_route', 'runner', 'phase',
                'requires_decision', 'numerical_solver'):
        parent[key] = parent_recipe[key]
    context.validate(parent, require_bound=require_bound)
    return {key: cfg[key] for key in ('adaptation_schema', 'adaptation_route', 'input_id',
            'graph_schedule', 'context_window_new_nodes', 'promotion_schedule',
            'early_numerical_solver', 'late_numerical_solver')}


def write_recipes(directory):
    import json
    from pathlib import Path
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for key in ORDER:
        cfg = recipe(key); validate(cfg)
        (directory / (ROUTE + '--' + key + '.json')).write_text(json.dumps(cfg, indent=2)+'\n')

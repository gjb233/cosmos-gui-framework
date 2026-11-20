"""Compare published targets to upstream commands without executing annotation code.

This is a diagnostic, not a trajectory/split certification or launch gate override.
"""
import ast
import collections
import json
from pathlib import Path

ROOT = Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')

def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)

def commands(text):
    result = []
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError('non-call annotation')
        call = node.value
        name = ast.unparse(call.func)
        args = [ast.literal_eval(x) for x in call.args]
        kwargs = {x.arg: ast.literal_eval(x.value) for x in call.keywords}
        if None in kwargs:
            raise ValueError('keyword expansion')
        result.append((name, args, kwargs))
    return result

def compare(action, calls):
    kind, value, point = action['action_type'], action.get('value'), action.get('point_2d')
    if not calls:
        return 'missing_command'
    name, args, kw = calls[-1]
    def text(key):
        return kw.get(key, args[0] if args else None)
    def coordinate():
        if not isinstance(point, list) or len(point) != 2:
            return False
        x, y = kw.get('x'), kw.get('y')
        # Both upstream rounded 0..1 and published integer 0..1000 coordinates
        # lose precision; tolerance is diagnostic only, never a repair.
        return x is not None and y is not None and max(abs(point[0]-1000*x), abs(point[1]-1000*y)) <= 2
    mapping = {
        'mobile.home': 'NavigateHome', 'mobile.back': 'NavigateBack',
        'mobile.terminate': 'Terminate', 'mobile.wait': 'Wait',
        'mobile.open_app': 'OpenApp', 'pyautogui.press': 'KeyboardPress',
        'pyautogui.click': 'Click', 'mobile.long_press': 'LongPress',
        'pyautogui.write': 'Write', 'answer': 'Answer',
    }
    if name in mapping:
        if kind != mapping[name]:
            return 'action_type_conflict'
        if name in ('pyautogui.click', 'mobile.long_press'):
            return 'coordinate_compatible' if coordinate() else 'coordinate_conflict'
        field = {'pyautogui.write': 'message', 'answer': 'answer',
                 'mobile.open_app': 'app_name', 'mobile.wait': 'seconds',
                 'pyautogui.press': 'key'}.get(name)
        if field:
            expected = text(field)
            if value != expected:
                return 'value_conflict'
        return 'command_compatible'
    if name == 'pyautogui.scroll':
        amount = kw.get('page', args[0] if args else None)
        if kind != 'Scroll' or not isinstance(amount, (int, float)) or amount == 0:
            return 'action_type_conflict'
        return 'command_compatible' if value == ('down' if amount < 0 else 'up') else 'direction_conflict'
    # Swipe coordinate systems require source-specific original annotation checks.
    if name == 'mobile.swipe':
        return 'requires_source_swipe_mapping' if kind in ('Swipe', 'Scroll') else 'action_type_conflict'
    return 'unsupported_upstream_command'

def main():
    actions = {x['id']: x for x in rows(ROOT/'artifacts/action-pair-audit.jsonl')}
    counts = collections.defaultdict(collections.Counter)
    output = ROOT/'artifacts/upstream-action-alignment.jsonl'
    with output.open('w') as stream:
        for join in rows(ROOT/'artifacts/upstream-annotation-joins.jsonl'):
            target = actions[join['id']]
            results = []
            for match in join['matches']:
                codes = [c['value'] for c in match['conversations'] if c.get('recipient') == 'os']
                try:
                    calls = [call for code in codes for call in commands(code)]
                    status = compare(target['action'], calls) if target.get('parsed') else 'target_parse_error'
                    results.append({'row_index': match['row_index'], 'status': status,
                                    'commands': codes})
                except (SyntaxError, ValueError, TypeError, KeyError) as exc:
                    results.append({'row_index': match['row_index'], 'status': 'unparsed_command',
                                    'error': str(exc), 'commands': codes})
            statuses = sorted({r['status'] for r in results})
            status = statuses[0] if len(statuses) == 1 else ('missing_upstream' if not statuses else 'ambiguous_results')
            counts[join['source']][status] += 1
            stream.write(json.dumps({'id': join['id'], 'source': join['source'],
                                     'status': status, 'matches': results,
                                     'original_trajectory_verified': False})+'\n')
    report = {'by_source': dict(counts), 'training_ready': False,
              'scope': 'syntactic command comparison only; original split, history, semantics and transitions still required'}
    (ROOT/'artifacts/upstream-action-alignment-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

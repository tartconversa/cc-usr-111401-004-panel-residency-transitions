"""连续时间多状态分析：数学正确性、口径、诊断与资料身份。"""
import json
import math
import random
import unittest
from pathlib import Path

from app import ctmc
from app.catalog import load
from app.mservice import SCHEMAS, Scheme, build_histories, run
from app.service import inspect

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample.json').read_text(encoding='utf-8'))


def close_matrix(A, B, tol=1e-9):
    return all(abs(a - b) <= tol for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def simulate_payload(seed=7, n_persons=150):
    """按已知强度模拟面板观察（固定随访日程，部分缺席）。"""
    states = ['短住', '常驻', '离村']
    edges = [['短住', '常驻'], ['常驻', '短住'], ['短住', '离村'], ['离村', '短住'], ['常驻', '离村']]
    q_true = [0.02, 0.01, 0.005, 0.03, 0.003]
    model = ctmc.PanelModel(3, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)], [0])
    theta = [math.log(q) for q in q_true]
    P = model.transition(theta, 0.0, 40.0)
    rng = random.Random(seed)
    visits = []
    for p in range(n_persons):
        state = 0 if rng.random() < 0.6 else 1
        for k, day in enumerate(range(0, 201, 40)):
            if k > 0:
                draw = rng.random()
                acc = 0.0
                for nxt, prob in enumerate(P[state]):
                    acc += prob
                    if draw <= acc:
                        state = nxt
                        break
            if k > 0 and rng.random() < 0.1:
                continue
            visits.append({'id': f'v{p}_{k}', 'revision': 1, 'person': f'p{p}',
                           'day': day, 'states': [states[state]]})
    payload = {'schema_version': 1,
               'metadata': {'data_version': 'sim', 'state_names': states,
                            'allowed_edges': edges, 'cutpoints': [0]},
               'collections': {'visits': visits}}
    return payload, q_true


class MatrixTests(unittest.TestCase):
    def test_expm_known_matrices(self):
        nilpotent = ctmc.mat_expm([[0.0, 1.0], [0.0, 0.0]])
        self.assertTrue(close_matrix(nilpotent, [[1.0, 1.0], [0.0, 1.0]]))
        diag = ctmc.mat_expm([[-1.0, 0.0], [0.0, -2.0]])
        self.assertTrue(close_matrix(diag, [[math.exp(-1), 0.0], [0.0, math.exp(-2)]]))

    def test_generator_expm_rows_sum_to_one(self):
        model = ctmc.PanelModel(3, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)], [0])
        theta = [math.log(q) for q in (0.02, 0.01, 0.005, 0.03, 0.003)]
        P = model.transition(theta, 0.0, 137.0)
        for row in P:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            self.assertTrue(all(x >= -1e-12 for x in row))

    def test_transition_matches_two_state_closed_form(self):
        a, b = 0.05, 0.02
        model = ctmc.PanelModel(2, [(0, 1), (1, 0)], [0])
        theta = [math.log(a), math.log(b)]
        lam = a + b
        for t in (1.0, 10.0, 83.0):
            e = math.exp(-lam * t)
            expected = [[b / lam + a / lam * e, a / lam * (1 - e)],
                        [b / lam * (1 - e), a / lam + b / lam * e]]
            self.assertTrue(close_matrix(model.transition(theta, 0.0, t), expected, 1e-9))

    def test_piecewise_transition_is_time_ordered_product(self):
        model = ctmc.PanelModel(2, [(0, 1)], [0, 50])
        theta = [math.log(0.1), math.log(0.02)]
        manual = ctmc.mat_mul(
            ctmc.mat_expm(ctmc.mat_scale(model.generator(theta, 0), 40.0)),
            ctmc.mat_expm(ctmc.mat_scale(model.generator(theta, 1), 30.0)))
        self.assertTrue(close_matrix(model.transition(theta, 10.0, 80.0), manual, 1e-12))


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = 0.05, 0.02
        self.model = ctmc.PanelModel(2, [(0, 1), (1, 0)], [0])
        self.theta = [math.log(self.a), math.log(self.b)]

    def closed(self, i, j, t):
        lam = self.a + self.b
        e = math.exp(-lam * t)
        table = [[self.b / lam + self.a / lam * e, self.a / lam * (1 - e)],
                 [self.b / lam * (1 - e), self.a / lam + self.b / lam * e]]
        return table[i][j]

    def test_loglik_matches_manual_forward(self):
        history = [[(0.0, frozenset({0})), (10.0, frozenset({1})), (25.0, frozenset({1}))]]
        expected = math.log(self.closed(0, 1, 10.0)) + math.log(self.closed(1, 1, 15.0))
        self.assertAlmostEqual(self.model.loglik(self.theta, history), expected, places=9)

    def test_uncertain_observation_sums_allowed_states(self):
        def ll(states):
            history = [[(0.0, frozenset({0})), (10.0, states), (25.0, frozenset({1}))]]
            return self.model.loglik(self.theta, history)
        total = ll(frozenset({0, 1}))
        parts = ll(frozenset({0})), ll(frozenset({1}))
        self.assertAlmostEqual(math.exp(total), math.exp(parts[0]) + math.exp(parts[1]), places=12)

    def test_no_exposure_before_first_or_after_last_visit(self):
        history = [[(30.0, frozenset({0})), (45.0, frozenset({1}))]]
        expected = math.log(self.closed(0, 1, 15.0))
        self.assertAlmostEqual(self.model.loglik(self.theta, history), expected, places=9)

    def test_single_visit_person_contributes_nothing(self):
        payload, _ = simulate_payload(seed=11, n_persons=8)
        extra = {'id': 'solo1', 'revision': 1, 'person': 'solo', 'day': 77, 'states': ['短住']}
        snap1 = load(payload, SCHEMAS)
        spec = Scheme(snap1.metadata, {})
        h1, _, singles1, _, _ = build_histories(snap1, spec)
        payload2 = json.loads(json.dumps(payload))
        payload2['collections']['visits'].append(extra)
        snap2 = load(payload2, SCHEMAS)
        h2, _, singles2, _, _ = build_histories(snap2, spec)
        self.assertEqual(singles2, singles1 + ['solo'])
        model = ctmc.PanelModel(3, spec.edges, spec.cutpoints)
        theta = [math.log(0.01)] * model.n_params
        self.assertEqual(model.loglik(theta, h1), model.loglik(theta, h2))


class SchemeTests(unittest.TestCase):
    def test_forbidden_edges_remove_transitions(self):
        snap = load(SAMPLE, SCHEMAS)
        spec = Scheme(snap.metadata, {'forbidden_edges': [['离村', '短住']]})
        self.assertNotIn((2, 0), spec.edges)
        model = ctmc.PanelModel(3, spec.edges, spec.cutpoints)
        theta = [-3.0] * model.n_params
        Q = model.generator(theta, 0)
        self.assertEqual(Q[2][0], 0.0)
        self.assertEqual(Q[2][2], 0.0)  # 离村无出路时该行全零，由配置决定而非程序特判

    def test_forbidden_edge_must_be_allowed(self):
        snap = load(SAMPLE, SCHEMAS)
        with self.assertRaisesRegex(ValueError, '禁止转移'):
            Scheme(snap.metadata, {'forbidden_edges': [['常驻', '离村']]})

    def test_empty_edges_rejected(self):
        snap = load(SAMPLE, SCHEMAS)
        with self.assertRaisesRegex(ValueError, '为空'):
            Scheme(snap.metadata, {'allowed_edges': []})


class FitDiagnosticsTests(unittest.TestCase):
    def test_recovers_simulated_intensities(self):
        payload, q_true = simulate_payload()
        result = run(payload, {'optimizer': {'max_iter': 1500, 'restarts': 2}})
        self.assertEqual(result['status'], 'ok', result['warnings'])
        params = result['fit']['parameters']
        self.assertEqual(len(params), len(q_true))
        # 极大似然估计的对数似然不应低于真实参数处
        snap = load(payload, SCHEMAS)
        spec = Scheme(snap.metadata, {})
        histories, _, _, _, _ = build_histories(snap, spec)
        model = ctmc.PanelModel(3, spec.edges, spec.cutpoints)
        ll_true = model.loglik([math.log(q) for q in q_true], histories)
        self.assertGreaterEqual(result['fit']['log_likelihood'], ll_true - 1e-6)
        for param, q in zip(params, q_true):
            tolerance = max(0.4, 3.0 * (param['se_log_intensity'] or 0.0))
            self.assertAlmostEqual(param['log_intensity'], math.log(q), delta=tolerance,
                                   msg=param['parameter'])
        back = next(p for p in params if p['parameter'] == 'seg0:离村->短住')
        self.assertGreater(back['intensity_per_day'], 0.0)  # 离村不是吸收终点
        self.assertTrue(all(p['identifiable'] for p in params))

    def test_boundary_solution_is_flagged_not_precise(self):
        states = ['短住', '常驻', '离村']
        edges = [['短住', '常驻'], ['常驻', '短住'], ['短住', '离村'], ['离村', '短住']]
        visits = []
        for p in range(8):
            for k, day in enumerate((0, 60, 120)):
                visits.append({'id': f'v{p}_{k}', 'revision': 1, 'person': f'p{p}',
                               'day': day, 'states': ['短住']})
        payload = {'schema_version': 1,
                   'metadata': {'state_names': states, 'allowed_edges': edges, 'cutpoints': [0]},
                   'collections': {'visits': visits}}
        result = run(payload)
        boundary = result['diagnostics']['boundary_parameters']
        self.assertIn('seg0:短住->常驻', boundary)
        self.assertIn('seg0:短住->离村', boundary)
        unidentifiable = result['diagnostics']['unidentifiable_parameters']
        self.assertIn('seg0:常驻->短住', unidentifiable)
        self.assertIn('seg0:离村->短住', unidentifiable)
        for param in result['fit']['parameters']:
            if param['boundary'] is not None or not param['identifiable']:
                self.assertIsNone(param['se_log_intensity'])
        self.assertTrue(any('边界' in w for w in result['warnings']))

    def test_non_convergence_is_not_disguised(self):
        payload, _ = simulate_payload(seed=13, n_persons=10)
        result = run(payload, {'optimizer': {'max_iter': 1, 'restarts': 1}})
        self.assertEqual(result['status'], 'not_converged')
        self.assertFalse(result['fit']['convergence']['converged'])
        self.assertEqual(result['prediction']['status'], 'not_converged')
        self.assertTrue(any('未收敛' in w for w in result['warnings']))


class ProvenanceAndSupportTests(unittest.TestCase):
    def test_provenance_binds_inputs_and_scheme(self):
        first = run(SAMPLE)
        self.assertEqual(first['input_digest'], inspect(SAMPLE)['input_digest'])
        self.assertEqual(first['scheme_digest'], run(SAMPLE)['scheme_digest'])
        other = run(SAMPLE, {'forbidden_edges': [['离村', '短住']]})
        self.assertNotEqual(first['scheme_digest'], other['scheme_digest'])
        self.assertEqual(first['prediction']['input_digest'], first['input_digest'])
        self.assertEqual(first['prediction']['scheme_digest'], first['scheme_digest'])

    def test_span_support_links_spans_to_parameters(self):
        payload = json.loads(json.dumps(SAMPLE))
        payload['collections']['visits'] = [
            {'id': 'a1', 'revision': 1, 'person': 'a', 'day': 30, 'states': ['短住']},
            {'id': 'a2', 'revision': 1, 'person': 'a', 'day': 120, 'states': ['常驻']},
        ]
        result = run(payload, {'cutpoints': [0, 90, 180]})
        self.assertEqual(len(result['span_support']), 1)
        support = result['span_support'][0]
        self.assertEqual(support['segments'], [0, 1])
        self.assertIn('seg0:短住->常驻', support['informed_parameters'])
        self.assertIn('seg1:短住->常驻', support['informed_parameters'])
        self.assertEqual(set(support['filtered_probabilities']), {'短住', '常驻', '离村'})
        names = {p['parameter'] for p in result['fit']['parameters']}
        self.assertTrue(set(support['informed_parameters']) <= names)

    def test_prediction_grid_and_extrapolation_flags(self):
        payload, _ = simulate_payload(seed=17, n_persons=12)
        result = run(payload)
        prediction = result['prediction']
        self.assertEqual(prediction['from_day'], 200.0)
        days = [g['day'] for g in prediction['grid']]
        self.assertEqual(days, sorted(days))
        for entry in prediction['grid']:
            self.assertAlmostEqual(sum(entry['probabilities'].values()), 1.0, places=9)
            occ = sum(entry['probabilities'][s] for s in prediction['occupancy_states'])
            self.assertAlmostEqual(entry['in_village'], occ, places=9)
            self.assertEqual(entry['extrapolated'], entry['day'] > 200.0)
        self.assertIn('离村', result['model']['states'])
        self.assertNotIn('离村', prediction['occupancy_states'])

    def test_assumptions_state_ignorability(self):
        result = run(SAMPLE)
        text = ' '.join(result['assumptions'])
        self.assertIn('观察时点可忽略', text)
        self.assertIn('稀疏观察不等于稳定留驻', text)


if __name__ == '__main__':
    unittest.main()

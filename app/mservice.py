"""连续时间多状态面板分析：拟合、预测、诊断与资料身份绑定。

在既有访问明细（app.baseline）之上增加连续时间马尔可夫模型：
状态之间按允许转移边往返，离村不自动成为吸收终点；观察间隔内
对所有允许的中间路径求和。结果通过input_digest与scheme_digest
绑定同一版观察和强度方案；资料修订后digest变化即另起一版。
"""
import hashlib
import json
import math
import sys
from pathlib import Path

from . import ctmc
from .baseline import analyze, visits_by_person
from .catalog import canonical, load
from .service import SCHEMAS

DEFAULT_STEP_DAYS = 30.0
DEFAULT_HORIZON_DAYS = 180.0
MAX_GRID_POINTS = 500
OUT_OF_VILLAGE = '离村'

ASSUMPTIONS = [
    '观察时点可忽略：假定访视时间安排不携带潜在状态过程的信息；若离村者更难被访到，估计会有偏。',
    '稀疏观察不等于稳定留驻：两次访问之间可能发生未观察到的往返；模型对所有允许的中间路径求和，'
    '但强度反映的是观察间隔内的净流动，不能证明期间持续留驻。',
    '分段恒定强度：每个分段内转移强度恒定，分段边界由cutpoints给定。',
    '条件入组：首访之前不计暴露，末访之后不补写状态；首访为集合时对集合内状态求和，不估计入组分布。',
    '“无法确定”按允许状态集合处理，不是额外的人生状态。',
]


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field}应为有限数')
    return float(value)


class Scheme:
    """规范化后的强度方案：分段边界、允许/禁止转移、在村状态、预测与优化配置。"""

    def __init__(self, metadata, raw):
        if not isinstance(raw, dict):
            raise ValueError('强度方案应为对象')
        state_names = metadata.get('state_names')
        if (not isinstance(state_names, list) or not state_names
                or any(not isinstance(s, str) or not s for s in state_names)
                or len(set(state_names)) != len(state_names)):
            raise ValueError('metadata.state_names应为非空不重复字符串数组')
        self.state_names = list(state_names)
        self.state_index = {s: i for i, s in enumerate(self.state_names)}
        self.cutpoints = self._cutpoints(metadata, raw)
        self.edges, self.edge_names = self._edges(metadata, raw)
        self.occupancy = self._occupancy(raw)
        self.occupancy_index = [self.state_index[s] for s in self.occupancy]
        self.prediction = self._prediction(raw)
        self.optimizer = self._optimizer(raw)
        self.json = {
            'cutpoints': self.cutpoints,
            'allowed_edges': [list(e) for e in self.edge_names],
            'occupancy_states': list(self.occupancy),
            'prediction': self.prediction,
            'optimizer': self.optimizer,
        }
        self.digest = hashlib.sha256(canonical(self.json).encode('utf-8')).hexdigest()

    def _cutpoints(self, metadata, raw):
        cutpoints = raw.get('cutpoints', metadata.get('cutpoints', [0]))
        if not isinstance(cutpoints, list) or not cutpoints:
            raise ValueError('cutpoints应为非空数组')
        points = [_number(c, 'cutpoints') for c in cutpoints]
        if any(b <= a for a, b in zip(points, points[1:])):
            raise ValueError('cutpoints应严格递增')
        return points

    def _edge_pair(self, value, field):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(s not in self.state_index for s in value)):
            raise ValueError(f'{field}应为两个已知状态名')
        if value[0] == value[1]:
            raise ValueError(f'{field}不允许自环')
        return self.state_index[value[0]], self.state_index[value[1]]

    def _edges(self, metadata, raw):
        allowed = raw.get('allowed_edges', metadata.get('allowed_edges', []))
        if not isinstance(allowed, list):
            raise ValueError('allowed_edges应为数组')
        edges, names = [], []
        for value in allowed:
            pair = self._edge_pair(value, 'allowed_edges')
            if pair in edges:
                raise ValueError('allowed_edges存在重复边')
            edges.append(pair)
            names.append((value[0], value[1]))
        forbidden = raw.get('forbidden_edges', [])
        if not isinstance(forbidden, list):
            raise ValueError('forbidden_edges应为数组')
        blocked = set()
        for value in forbidden:
            pair = self._edge_pair(value, 'forbidden_edges')
            if pair not in edges:
                raise ValueError('禁止转移不在允许集合内')
            blocked.add(pair)
        kept = [(pair, name) for pair, name in zip(edges, names) if pair not in blocked]
        if not kept:
            raise ValueError('允许转移集合为空')
        return [pair for pair, _ in kept], [name for _, name in kept]

    def _occupancy(self, raw):
        occupancy = raw.get('occupancy_states')
        if occupancy is None:
            occupancy = [s for s in self.state_names if s != OUT_OF_VILLAGE] or list(self.state_names)
        if (not isinstance(occupancy, list) or not occupancy
                or any(s not in self.state_index for s in occupancy)
                or len(set(occupancy)) != len(occupancy)):
            raise ValueError('occupancy_states应为已知状态的非空不重复数组')
        return list(occupancy)

    def _prediction(self, raw):
        spec = raw.get('prediction', {})
        if not isinstance(spec, dict):
            raise ValueError('prediction应为对象')
        from_day = spec.get('from_day')
        to_day = spec.get('to_day')
        step = spec.get('step', DEFAULT_STEP_DAYS)
        step = _number(step, 'prediction.step')
        if step <= 0:
            raise ValueError('prediction.step应为正数')
        start = spec.get('start')
        if start is not None:
            if not isinstance(start, dict):
                raise ValueError('prediction.start应为对象')
            if 'state' in start:
                if start['state'] not in self.state_index:
                    raise ValueError('prediction.start.state应为已知状态')
                start = {'state': start['state']}
            elif 'distribution' in start:
                dist = start['distribution']
                if (not isinstance(dist, dict) or not dist
                        or any(s not in self.state_index for s in dist)):
                    raise ValueError('prediction.start.distribution应引用已知状态')
                weights = {s: _number(w, 'prediction.start.distribution') for s, w in dist.items()}
                if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
                    raise ValueError('prediction.start.distribution权重应为非负且和为正')
                start = {'distribution': weights}
            else:
                raise ValueError('prediction.start应给出state或distribution')
        return {
            'from_day': None if from_day is None else _number(from_day, 'prediction.from_day'),
            'to_day': None if to_day is None else _number(to_day, 'prediction.to_day'),
            'step': step,
            'start': start,
        }

    def _optimizer(self, raw):
        spec = raw.get('optimizer', {})
        if not isinstance(spec, dict):
            raise ValueError('optimizer应为对象')
        max_iter = spec.get('max_iter', 2000)
        restarts = spec.get('restarts', 3)
        for name, value in (('max_iter', max_iter), ('restarts', restarts)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'optimizer.{name}应为正整数')
        return {'max_iter': max_iter, 'restarts': restarts}


def build_histories(snapshot, scheme):
    """把有效访问整理成模型用面板历史。

    返回(参与似然的历史, 对应人名, 仅单次访问的人, 每人末次允许集合, 全部观察日)。
    仅单次访问的人只贡献入组条件，不进入似然；首访之前与末访之后都不计暴露。
    """
    people = visits_by_person(snapshot)
    histories, persons, singles, last_sets, all_days = [], [], [], [], []
    for person in sorted(people):
        days = people[person]
        ordered = sorted(days)
        all_days.extend(float(d) for d in ordered)
        obs = [(float(day), frozenset(scheme.state_index[s] for s in days[day]))
               for day in ordered]
        last_sets.append(obs[-1][1])
        if len(obs) >= 2:
            persons.append(person)
            histories.append(obs)
        else:
            singles.append(person)
    return histories, persons, singles, last_sets, all_days


def _segments_crossed(model, a, b):
    out = []
    for seg in range(model.segment_of(a), model.n_segments):
        lo, hi = model.segment_bounds(seg)
        if min(b, hi) > max(a, lo):
            out.append(seg)
    return out


def _param_name(scheme, seg, edge):
    i, j = scheme.edges[edge]
    return f'seg{seg}:{scheme.state_names[i]}->{scheme.state_names[j]}'


def _start_distribution(scheme, last_sets):
    start = scheme.prediction['start']
    n = len(scheme.state_names)
    if start is None:
        weights = [0.0] * n
        for states in last_sets:
            for s in states:
                weights[s] += 1.0 / len(states)
        kind = 'last_observation_empirical'
    elif 'state' in start:
        weights = [0.0] * n
        weights[scheme.state_index[start['state']]] = 1.0
        kind = 'state'
    else:
        weights = [0.0] * n
        for name, w in start['distribution'].items():
            weights[scheme.state_index[name]] = w
        kind = 'distribution'
    total = sum(weights)
    weights = [w / total for w in weights]
    description = {scheme.state_names[i]: weights[i] for i in range(n) if weights[i] > 0.0}
    return weights, {'type': kind, 'distribution': description}


def _predict(model, scheme, theta, last_sets, all_days, cov, status, digest_pair):
    from_day = scheme.prediction['from_day']
    if from_day is None:
        from_day = max(all_days)
    to_day = scheme.prediction['to_day']
    if to_day is None:
        to_day = from_day + DEFAULT_HORIZON_DAYS
    if to_day < from_day:
        raise ValueError('prediction.to_day早于from_day')
    step = scheme.prediction['step']
    count = int(math.floor((to_day - from_day) / step + 1e-9)) + 1
    if count > MAX_GRID_POINTS:
        raise ValueError('预测网格点数超出上限')
    days = [from_day + k * step for k in range(count)]
    start_vec, start_desc = _start_distribution(scheme, last_sets)
    occ = scheme.occupancy_index

    def in_village(point_theta):
        cache = {}
        values = []
        for day in days:
            P = model.transition(point_theta, from_day, day, cache)
            probs = ctmc.mat_vec(start_vec, P)
            total = sum(probs)
            values.append(sum(probs[i] for i in occ) / total)
        return values

    base = in_village(theta)
    ses = [None] * len(days)
    if cov is not None:
        grads = [[] for _ in days]
        for pos in range(model.n_params):
            h = 1e-5 * (1.0 + abs(theta[pos]))
            tp, tm = list(theta), list(theta)
            tp[pos] += h
            tm[pos] -= h
            hi, lo = in_village(tp), in_village(tm)
            for d in range(len(days)):
                grads[d].append((hi[d] - lo[d]) / (2.0 * h))
        for d in range(len(days)):
            var = 0.0
            for i in range(model.n_params):
                for j in range(model.n_params):
                    var += grads[d][i] * cov[i][j] * grads[d][j]
            ses[d] = math.sqrt(max(var, 0.0))
    lo_day, hi_day = min(all_days), max(all_days)
    cache = {}
    grid = []
    for k, day in enumerate(days):
        P = model.transition(theta, from_day, day, cache)
        probs = ctmc.mat_vec(start_vec, P)
        total = sum(probs)
        grid.append({
            'day': day,
            'probabilities': {scheme.state_names[i]: probs[i] / total for i in range(model.n_states)},
            'in_village': base[k],
            'in_village_se': ses[k],
            'extrapolated': day < lo_day or day > hi_day,
        })
    return {
        'from_day': from_day, 'to_day': to_day, 'step': step, 'start': start_desc,
        'occupancy_states': list(scheme.occupancy),
        'grid': grid,
        'status': status,
        'input_digest': digest_pair[0],
        'scheme_digest': digest_pair[1],
    }


def run(payload, scheme=None):
    """拟合连续时间多状态模型并给出预测、诊断与逐跨度支持。

    输出绑定input_digest（同一版观察）与scheme_digest（同一套强度方案）。
    边界解、未收敛与不可识别参数都显式标记，不伪装成精确估计。
    """
    snapshot = load(payload, SCHEMAS)
    metadata = snapshot.metadata
    spec = Scheme(metadata, scheme or {})
    histories, persons, singles, last_sets, all_days = build_histories(snapshot, spec)
    spans = analyze(snapshot)
    model = ctmc.PanelModel(len(spec.state_names), spec.edges, spec.cutpoints)
    digests = (snapshot.digest, spec.digest)
    result = {
        'schema_version': 1,
        'analysis': 'ctmc_panel',
        'input_digest': snapshot.digest,
        'data_version': metadata.get('data_version'),
        'scheme_digest': spec.digest,
        'scheme': spec.json,
        'model': {
            'states': spec.state_names,
            'segments': [
                {'index': k,
                 'start': spec.cutpoints[k],
                 'end': spec.cutpoints[k + 1] if k + 1 < len(spec.cutpoints) else None}
                for k in range(len(spec.cutpoints))
            ],
            'edges': [{'from': a, 'to': b} for a, b in spec.edge_names],
            'occupancy_states': list(spec.occupancy),
            'absorbing_note': '离村不自动视为吸收终点；是否可返回完全由允许转移边决定。',
        },
        'assumptions': list(ASSUMPTIONS),
    }
    warnings = []
    diagnostics = {
        'data_range': [min(all_days), max(all_days)] if all_days else None,
        'n_persons': len(persons) + len(singles),
        'n_persons_in_likelihood': len(persons),
        'single_visit_persons': singles,
        'n_spans': len(spans),
    }
    if singles:
        warnings.append(f'{len(singles)}人仅有一次访问，只贡献入组条件，不进入似然')
    if not histories:
        result.update(status='failed', fit=None, span_support=[], prediction=None,
                      diagnostics=diagnostics,
                      warnings=warnings + ['没有任何人具有两次及以上访问，无法拟合强度'])
        return result

    intervals = [obs[k + 1][0] - obs[k][0] for obs in histories for k in range(len(obs) - 1)]
    mean_dt = sum(intervals) / len(intervals)
    max_interval = max(intervals)
    theta0 = [math.log(1.0 / mean_dt)] * model.n_params
    opt = spec.optimizer
    fitted = ctmc.fit(model, histories, theta0,
                      max_iter=opt['max_iter'], restarts=opt['restarts'])
    if fitted['failed']:
        result.update(status='failed', fit=None, span_support=[], prediction=None,
                      diagnostics=diagnostics,
                      warnings=warnings + ['似然在所试初值下不可用，拟合失败，不输出估计'])
        return result
    theta = fitted['theta']
    loglik = model.loglik(theta, histories)
    status = 'ok' if fitted['converged'] else 'not_converged'
    if not fitted['converged']:
        warnings.append('优化未收敛：以下估计与预测为临时结果，不能当作精确估计')

    # 每个参数的数据支持：覆盖该分段的随访区间条数与总天数
    support = [[0, 0.0] for _ in range(model.n_params)]
    for obs in histories:
        for k in range(1, len(obs)):
            a, b = obs[k - 1][0], obs[k][0]
            for seg in _segments_crossed(model, a, b):
                lo, hi = model.segment_bounds(seg)
                overlap = min(b, hi) - max(a, lo)
                for edge in range(len(spec.edges)):
                    pos = model.index[(seg, edge)]
                    support[pos][0] += 1
                    support[pos][1] += overlap

    objective = lambda t: ctmc.negative_loglik(model, histories, t)
    info = ctmc.hessian(objective, theta)
    grad = ctmc.gradient(objective, theta)
    eigen = ctmc.sym_eigenvalues(info)
    cov = None
    if eigen[0] > 1e-10 * max(eigen[-1], 1e-300):
        cov = ctmc.mat_inverse(info)
    if cov is None:
        warnings.append('观察信息矩阵非正定或接近奇异：标准误与预测不确定度不可用')
    max_curvature = max((info[i][i] for i in range(model.n_params)), default=0.0)

    def likelihood_range(pos):
        """在整个值域上探测：对数似然随该参数的最大变化。

        全程几乎不变说明数据不含该参数信息（不可识别）；在边界附近
        平坦但向内部显著下降则是边界解，两者不能混为一谈。
        """
        probes = (math.log(1e-6), math.log(1e-3), math.log(1.0 / mean_dt), 0.0)
        values = []
        for probe in probes:
            point = list(theta)
            point[pos] = probe
            values.append(ctmc.negative_loglik(model, histories, point))
        return max(values) - min(values)

    parameters = []
    boundary_parameters = []
    unidentifiable = []
    for pos, (seg, edge) in enumerate(model.params):
        q = math.exp(theta[pos])
        spans_n, days_n = support[pos]
        boundary = None
        if theta[pos] <= ctmc.LOG_LOWER + 1.0 or (q * max_interval < 1e-4 and grad[pos] > 0):
            boundary = 'lower'
        elif theta[pos] >= ctmc.LOG_UPPER - 1.0 or (q * mean_dt > 10.0 and grad[pos] < 0):
            boundary = 'upper'
        curvature = info[pos][pos]
        ll_range = likelihood_range(pos)
        insensitive = ll_range < 1e-4
        identifiable = boundary is None and days_n > 0 and not insensitive
        se_log = None
        if cov is not None and boundary is None and cov[pos][pos] >= 0.0:
            se_log = math.sqrt(cov[pos][pos])
        name = _param_name(spec, seg, edge)
        i, j = spec.edges[edge]
        parameters.append({
            'parameter': name,
            'segment': seg,
            'from': spec.state_names[i],
            'to': spec.state_names[j],
            'log_intensity': theta[pos],
            'intensity_per_day': q,
            'se_log_intensity': se_log,
            'boundary': boundary,
            'curvature': curvature,
            'likelihood_range': ll_range,
            'support_spans': spans_n,
            'support_days': days_n,
            'identifiable': identifiable,
        })
        if days_n <= 0.0:
            unidentifiable.append(name)
            warnings.append(f'参数{name}所在分段无任何随访区间覆盖，不可识别')
        elif insensitive:
            unidentifiable.append(name)
            warnings.append(f'参数{name}在整条值域上几乎不改变对数似然，数据不含其信息，不可识别')
        elif boundary is not None:
            boundary_parameters.append(name)
            side = '下界（与零无法区分）' if boundary == 'lower' else '上界'
            warnings.append(f'参数{name}触及强度{side}：按边界解报告，标准误不适用')
    if boundary_parameters and cov is not None:
        warnings.append('存在边界参数：其余标准误按条件于边界值理解，不确定度偏乐观')

    # 逐跨度支持：把既有访问明细接到拟合结果上
    records = []
    model.loglik(theta, histories, record=records)
    span_map = {(s['person'], float(s['start']), float(s['end'])): s for s in spans}
    span_support = []
    for rec in records:
        person = persons[rec['person_index']]
        span = span_map[(person, rec['start'], rec['end'])]
        segs = _segments_crossed(model, rec['start'], rec['end'])
        span_support.append({
            'person': person,
            'start': span['start'],
            'end': span['end'],
            'from_states': span['from_states'],
            'to_states': span['to_states'],
            'log_likelihood': rec['log_likelihood'],
            'segments': segs,
            'informed_parameters': [_param_name(spec, seg, edge)
                                    for seg in segs for edge in range(len(spec.edges))],
            'filtered_probabilities': {spec.state_names[i]: p
                                       for i, p in enumerate(rec['filtered'])},
        })

    diagnostics.update({
        'boundary_parameters': boundary_parameters,
        'unidentifiable_parameters': unidentifiable,
        'min_eigenvalue': eigen[0],
        'max_eigenvalue': eigen[-1],
        'covariance_available': cov is not None,
    })
    fit_out = {
        'log_likelihood': loglik,
        'n_parameters': model.n_params,
        'parameters': parameters,
        'convergence': {
            'converged': fitted['converged'],
            'evaluations': fitted['evaluations'],
            'restarts': fitted['restarts'],
        },
    }
    prediction = _predict(model, spec, theta, last_sets, all_days, cov, status, digests)
    result.update(status=status, fit=fit_out, span_support=span_support,
                  prediction=prediction, diagnostics=diagnostics, warnings=warnings)
    return result


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('请指定资料JSON路径，可选强度方案JSON路径')
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    scheme = None
    if len(sys.argv) == 3:
        scheme = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    print(json.dumps(run(payload, scheme), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()

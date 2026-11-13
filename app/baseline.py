"""现有业务摘要，不推断未观察到的业务关系。"""
def analyze(snapshot):
    people = {}
    valid = set(snapshot.metadata['state_names'])
    for row in snapshot.rows('visits'):
        states = set(row['states'])
        if not states or not states <= valid:
            raise ValueError('未知或空观察状态')
        by_day = people.setdefault(row['person'], {})
        states &= by_day.get(row['day'], states)
        if not states:
            raise ValueError('同日观察状态冲突')
        by_day[row['day']] = states
    spans = []
    for person, days in sorted(people.items()):
        ordered = sorted(days)
        for a, b in zip(ordered, ordered[1:]):
            spans.append({'person': person, 'start': a, 'end': b,
                          'from_states': sorted(days[a]), 'to_states': sorted(days[b])})
    return spans

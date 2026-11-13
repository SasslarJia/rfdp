"""Admission schedules; bridge proposals always retain the original budget."""
from MODEL.Paper_RFDP.PaperFullIterative import FullIHyRDP

SCHEDULES = ('fixed5', 'early5', 'late5')


def promotion_step(schedule, promoted, base_step=5):
    """Count newly admitted graph nodes, never evaluation-valid images.

    late5 admits new nodes 101--105 one at a time. Before that window a
    shortened batch may be needed to land exactly at 100 admitted nodes.
    """
    if schedule not in SCHEDULES or base_step != 5 or promoted < 0:
        raise ValueError('Invalid fixed-five promotion schedule')
    if schedule == 'early5' and promoted < 5:
        return 1
    if schedule == 'late5':
        if promoted < 100:
            return min(5, 100 - promoted)
        if promoted < 105:
            return 1
    return 5


class ScheduledIHyRDP(FullIHyRDP):
    def __init__(self, *args, promotion_schedule='fixed5', **kwargs):
        promotion_step(promotion_schedule, 0, kwargs.get('step', 5))
        self.promotion_schedule = promotion_schedule
        super().__init__(*args, **kwargs)

    def bridges(self, active_components):
        current = self.step
        try:
            self.step = 5
            return super().bridges(active_components)
        finally:
            self.step = current

    def rank(self, seeds, nq=None, input_similarity=None, numerical_tail=False,
             bridge=True, trace=None, progress=None):
        promoted = 0
        self.step = promotion_step(self.promotion_schedule, promoted)

        def capture(record):
            nonlocal promoted
            record['promotion_step'] = self.step
            record['promotion_schedule'] = self.promotion_schedule
            if trace is not None:
                trace(record)
            promoted += len(record['selected'])
            self.step = promotion_step(self.promotion_schedule, promoted)

        try:
            return super().rank(seeds, nq=nq, input_similarity=input_similarity,
                                numerical_tail=numerical_tail, bridge=bridge,
                                trace=capture, progress=progress)
        finally:
            self.step = 5

"""The live buffer retains the latest detail per module and block."""
import json
from pathlib import Path
import tempfile
import unittest

from modern_pipeline.live_trace import LiveBuffer, LiveTrace, SAMPLING


def computation(node, calculations, model='stgcn', frame=1):
    return {'kind': 'computation', 'frame': frame, 'time_seconds': 0,
            'model': model, 'node': node, 'stage': 5, 'seq': 0,
            'calculations': calculations}


def yolo_row(module, detail=True):
    return {'id': 'yolo.conv.%d' % module, 'stage': 'yolo', 'module': module,
            'operation': 'dense_conv', 'inputs': [], 'weights': [], 'bias': 0.,
            'actual': None, 'metadata': {'detail': True} if detail else {}}


def stgcn_row(block, identity=None):
    return {'id': 'stgcn.spatial.block%d.track1%s' % (block, '' if identity is None else identity),
            'stage': 'spatial', 'inputs': [1.] * 51, 'weights': [1.] * 51,
            'bias': 0., 'actual': 51.,
            'metadata': {'detail': True, 'detail_block': block}}


class LiveDetailBufferTests(unittest.TestCase):
    def test_details_are_bounded_and_survive_snapshot(self):
        buffer = LiveBuffer('job-1')
        for module in range(5):
            buffer.append(computation('flow.yolo.conv.%d' % module,
                                      [yolo_row(module)], model='yolo'))
        for block in range(2):
            buffer.append(computation('flow.stgcn.spatial.block%d' % block,
                                      [stgcn_row(block)]))
        snapshot = buffer.snapshot('running')
        self.assertEqual(sorted(snapshot['details']['yolo']), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(snapshot['details']['stgcn']), [0, 1])
        self.assertEqual(snapshot['details']['yolo'][4]['id'], 'yolo.conv.4')
        self.assertEqual(len(snapshot['details']['stgcn'][1]), 1)
        self.assertLessEqual(len(snapshot['details']['yolo']), 24)
        self.assertLessEqual(len(snapshot['details']['stgcn']), 10)
        # Snapshots must not alias the live cache.
        snapshot['details']['yolo'][4]['id'] = 'changed'
        snapshot['details']['stgcn'][1].clear()
        fresh = buffer.snapshot('running')
        self.assertEqual(fresh['details']['yolo'][4]['id'], 'yolo.conv.4')
        self.assertEqual(len(fresh['details']['stgcn'][1]), 1)

    def test_detail_maps_keep_only_the_latest_modules_blocks_and_rows(self):
        buffer = LiveBuffer('job-2')
        for module in range(30):
            buffer.append(computation('flow.yolo.conv.%d' % module,
                                      [yolo_row(module)], model='yolo'))
        for block in range(12):
            rows = ([stgcn_row(block, '.%d' % number) for number in range(15)]
                    if block == 11 else [stgcn_row(block)])
            buffer.append(computation('flow.stgcn.spatial.block%d' % block, rows))
        snapshot = buffer.snapshot('running')
        self.assertEqual(sorted(snapshot['details']['yolo']), list(range(6, 30)))
        self.assertEqual(sorted(snapshot['details']['stgcn']), list(range(2, 12)))
        rows = snapshot['details']['stgcn'][11]
        self.assertEqual([row['id'].rsplit('.', 1)[-1] for row in rows],
                         [str(number) for number in range(3, 15)])

    def test_rows_without_detail_metadata_are_ignored(self):
        buffer = LiveBuffer('job-3')
        buffer.append(computation('flow.yolo.conv.0', [yolo_row(0, detail=False)],
                                  model='yolo'))
        buffer.append(computation('flow.stgcn.normalize', [
            {'id': 'lstm.normalize.track1', 'stage': 'normalization',
             'inputs': [], 'weights': [], 'metadata': {}}]))
        buffer.append(computation('flow.stgcn.spatial.block0', [
            {'id': 'stgcn.spatial.block0.track1', 'stage': 'spatial',
             'inputs': [], 'weights': [], 'metadata': {'detail': True}}]))
        self.assertEqual(buffer.snapshot('running')['details'],
                         {'yolo': {}, 'stgcn': {}})

    def test_saved_snapshot_restores_details(self):
        buffer = LiveBuffer('job-4')
        buffer.append(computation('flow.yolo.conv.7', [yolo_row(7)], model='yolo'))
        buffer.append(computation('flow.stgcn.spatial.block3', [stgcn_row(3)]))
        buffer.finish('completed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'live.json'
            buffer.save(path, 'completed')
            restored = LiveBuffer('job-4')
            restored.restore(path)
        snapshot = restored.snapshot('completed')
        self.assertEqual(sorted(snapshot['details']['yolo']), [7])
        self.assertEqual(sorted(snapshot['details']['stgcn']), [3])
        self.assertEqual(snapshot['details']['yolo'][7]['id'], 'yolo.conv.7')
        self.assertEqual(snapshot['details']['stgcn'][3][0]['id'],
                         'stgcn.spatial.block3.track1')
        # Appending after a restart must keep the restored integer keys.
        restored.append(computation('flow.yolo.conv.1', [yolo_row(1)], model='yolo'))
        self.assertEqual(sorted(restored.snapshot('running')['details']['yolo']), [1, 7])

    def test_default_ring_and_frame_budgets_match_sampling(self):
        buffer = LiveBuffer('job-5')
        self.assertEqual(SAMPLING['ring_max_events'], 1024)
        self.assertEqual(SAMPLING['ring_max_bytes'], 8 * 1024 * 1024)
        self.assertEqual(SAMPLING['frame_max_events'], 192)
        self.assertEqual(buffer.max_events, 1024)
        self.assertEqual(buffer.max_bytes, 8 * 1024 * 1024)


class LiveDetailCoverageTests(unittest.TestCase):
    def coverage(self, events):
        return [event for event in events if event['node'] == 'detail.coverage']

    def test_coverage_event_advertises_scheduled_modules_and_block(self):
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.yolo_module_count = 24
        try:
            trace.begin_frame(5, 0.5)
            trace.end_frame([])
            rows = self.coverage(events)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row['model'], row['stage'], row['frame']), ('pipeline', 1, 5))
            self.assertIsNone(row['duration_ms'])
            self.assertEqual(row['yolo_modules'], [0, 1, 2, 3])
            self.assertEqual(row['stgcn_block'], 0)
            self.assertEqual(row['detail_frame'], 1)
            self.assertIs(row['detail_boundary'], False)
            self.assertEqual(events[-1]['node'], 'pipeline.output')
            # Coverage is not detail arithmetic, so it must not carry detail.
            self.assertFalse(any(event.get('detail') and 'detail_block' not in event
                                 for event in events))
            events.clear()
            trace.begin_frame(6, 1.5)
            trace.end_frame([])
            row = self.coverage(events)[0]
            self.assertEqual(row['yolo_modules'], [])
            self.assertEqual(row['stgcn_block'], 0)
            self.assertEqual(row['detail_frame'], 2)
            self.assertIs(row['detail_boundary'], True)
            events.clear()
            trace.begin_frame(7, 2.5)
            trace.end_frame([])
            row = self.coverage(events)[0]
            self.assertEqual(row['yolo_modules'], [4, 5, 6, 7])
            self.assertEqual(row['stgcn_block'], 1)
            self.assertEqual(row['detail_frame'], 3)
        finally:
            trace.close()

    def test_coverage_is_emitted_only_on_detail_frames(self):
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=999)
        try:
            trace.begin_frame(1, 0)
            trace.end_frame([])
            trace.begin_frame(2, 0.1)
            self.assertFalse(trace.detail_frame)
            trace.end_frame([])
        finally:
            trace.close()
        rows = self.coverage(events)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['frame'], rows[0]['detail_frame']), (1, 1))

    def test_detail_off_never_emits_coverage(self):
        events = []
        trace = LiveTrace(events.append, interval=0)
        try:
            trace.begin_frame(1, 0)
            trace.end_frame([])
        finally:
            trace.close()
        self.assertEqual(self.coverage(events), [])

    def test_heavy_block_and_dense_yolo_pairing_schedules_two_modules(self):
        events = []
        trace = LiveTrace(events.append, interval=0, detail=True, detail_interval=0)
        trace.yolo_module_count = 24
        trace.stgcn_terms = [64 * 9] * 7 + [256 * 9] * 3
        trace.yolo_module_terms = {4: 576, 5: 2304, 6: 576, 7: 2304,
                                   12: 0, 13: 576, 14: 0, 15: 0}
        expected = ((14, 15, 7, {4, 5}),
                    (2, 3, 1, {4, 5, 6, 7}),
                    (18, 19, 9, {12, 13, 14, 15}))
        try:
            for preset, index, block, modules in expected:
                trace.detail_index = preset
                trace.begin_frame(index, index / 30.)
                self.assertEqual(trace.detail_index, index)
                self.assertEqual(trace.detail_block, block)
                self.assertEqual(trace.detail_modules, modules)
                trace.end_frame([])
        finally:
            trace.close()
        rows = self.coverage(events)
        self.assertEqual([set(row['yolo_modules']) for row in rows],
                         [modules for _, _, _, modules in expected])
        self.assertEqual([row['stgcn_block'] for row in rows],
                         [block for _, _, block, _ in expected])


if __name__ == '__main__':
    unittest.main()

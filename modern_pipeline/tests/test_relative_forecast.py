import unittest
import torch
from modern_pipeline.behavior import forecast_with_statistics
from modern_pipeline.tests import test_live_arithmetic


class RelativeForecastTests(unittest.TestCase):
    def test_each_round_uses_its_latest_pose_and_emits_anchor(self):
        class UnitDelta(torch.nn.Module):
            def forward(self, x):
                return x.new_ones((len(x), 10, 17, 3))
        poses = torch.arange(50.).view(1, 50, 1, 1).expand(2, 50, 17, 3).clone()
        poses[1] += 100
        std = torch.full((17, 3), 2.)
        events = []
        out = forecast_with_statistics(UnitDelta(), poses, torch.zeros(17, 3), std,
                                       events.append, 'last_observation_relative')
        for k, event in enumerate(events):
            torch.testing.assert_close(torch.from_numpy(event['anchor']), poses[:, -1:] + 2*k)
            torch.testing.assert_close(out[:, k*10:(k+1)*10], (poses[:, -1:] + 2*(k+1)).expand(-1,10,-1,-1))

    def test_live_arithmetic_reconstructs_all_relative_rounds(self):
        fixture = test_live_arithmetic.LiveArithmeticTests()
        predictor = fixture.predictor('training_stats')
        predictor.forecast_representation = 'last_observation_relative'
        poses = torch.randn(3, 50, 17, 3).numpy() * 10 + 100
        _, events, _ = fixture.capture(predictor, poses)
        calculations = [c for event in events for c in event.get('calculations', [])]
        rows = [c for c in calculations if c['stage'] in ('normalization','denormalization')]
        self.assertTrue(rows)
        for row in rows:
            expected = sum(a*b for a,b in zip(row['inputs'],row['weights'])) + row['bias']
            self.assertAlmostEqual(expected, row['actual'], delta=1e-3)


if __name__ == '__main__': unittest.main()

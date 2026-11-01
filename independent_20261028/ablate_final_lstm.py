"""Paired diagnostic of the frozen final bundle, with and without LSTM.

Both branches reuse identical cached tracker histories, labels, reference
poses and classifier weights. This is a component-removal diagnostic of the
deployed bundle, not retraining or an independent generalization estimate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import experiment as e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, references = e.causal_test('modern')
    weights = e.ROOT / 'weights/final'
    result = {
        'protocol': 'Paired frozen-bundle LSTM removal; identical histories, reference poses and normalization rule; no retraining',
        'scope': 'Overlapping-source diagnostic. Classifier training included windows from the video pool.',
        'n': len(data['keys']), 'eligible': data['eligible'],
        'coverage': len(data['keys']) / data['eligible'],
        'device': args.device,
        'normalization': 'legacy_pair',
        'weights_sha256': {n: hashlib.sha256((weights / n).read_bytes()).hexdigest() for n in ['stgcn.pt', 'lstm.pt']},
        'branches': {},
    }
    predictions = {}
    for branch, lstm in [('observed', None), ('forecast', weights / 'lstm.pt')]:
        model = e.BehaviorPredictor(weights / 'stgcn.pt', lstm, args.device, 'legacy_pair')
        pred = [None] * len(data['keys'])
        begin = time.perf_counter()
        for video in sorted(references):
            ids = [i for i, k in enumerate(data['keys']) if k[0] == video]
            for start in range(0, len(ids), 32):
                ix = ids[start:start + 32]
                labels = model.predict(data['x'][ix].numpy(), references[video])['class_ids']
                for i, label in zip(ix, labels):
                    pred[i] = int(label)
        assert all(p is not None for p in pred)
        result['branches'][branch] = e.metrics(data['y'].tolist(), pred)
        result['branches'][branch]['batch_evaluation_seconds'] = time.perf_counter() - begin
        predictions[branch] = pred
        print(branch, json.dumps(result['branches'][branch]), flush=True)
        del model
        e.torch.cuda.empty_cache()
    old = Path('D:/SRTP_presentation_20261029/evaluation/predictions.json')
    if old.exists():
        old_rows = json.loads(old.read_text(encoding='utf-8'))
        lookup = {(r['video'], r['person'], r['frame']): (r['truth'], r['prediction']) for r in old_rows}
        result['forecast_matches_previous_by_key'] = all(
            lookup.get(tuple(k)) == (int(y), p)
            for k, y, p in zip(data['keys'], data['y'], predictions['forecast'])) and len(lookup) == len(data['keys'])
        assert result['forecast_matches_previous_by_key']
    rows = [dict(video=k[0], person=k[1], frame=k[2], truth=int(y),
                 observed=predictions['observed'][i], forecast=predictions['forecast'][i])
            for i, (k, y) in enumerate(zip(data['keys'], data['y']))]
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output / 'predictions.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Downloads records' acceleration (g) and metadata from the NGA-gm API
(https://nga-gm.org) into data/<time_series_metadata_id>.npz.

    python code/fetch_records.py --token-file ~/token.txt 57394 74
    python code/fetch_records.py --token-file ~/token.txt --ids-file data/sample_ids.txt

The API needs a bearer token from a logged-in nga-gm.org account.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = 'https://nga-gm.org'


def get(path, token):
    request = urllib.request.Request(API + path, headers={'Accept': 'application/json',
                                                          'Authorization': 'Bearer ' + token})
    return json.load(urllib.request.urlopen(request, timeout=300))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ids', nargs='*', type=int, help='time_series_metadata_ids')
    parser.add_argument('--ids-file', help='a file of time_series_metadata_ids, one per line')
    parser.add_argument('--token-file', required=True)
    args = parser.parse_args()
    token = open(os.path.expanduser(args.token_file)).read().strip()
    ids = list(args.ids)
    if args.ids_file:
        ids += [int(line) for line in open(args.ids_file) if line.strip()]
    out = os.path.join(ROOT, 'data')
    os.makedirs(out, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(out, f'{i}.npz'))]
    for start in range(0, len(todo), 10):
        batch = todo[start:start + 10]
        where = urllib.parse.quote('time_series_metadata_id IN (' + ','.join(map(str, batch)) + ')')
        meta = {m['time_series_metadata_id']: m for m in get('/time-series-metadata?where=' + where, token)}
        series = get('/time-series-data?component=acc_h1,acc_h2,acc_v&time_series_metadata_id='
                     + ','.join(map(str, batch)), token)
        for record in series:
            tsm = record['time_series_metadata_id']
            m = meta.get(tsm, {})
            np.savez_compressed(
                os.path.join(out, f'{tsm}.npz'),
                dt=1 / float(m.get('sampling_rate') or record['sampling_rate']),
                motion_id=m.get('motion_id', -1),
                azimuth_h1=m.get('azimuth_h1') if m.get('azimuth_h1') is not None else np.nan,
                azimuth_h2=m.get('azimuth_h2') if m.get('azimuth_h2') is not None else np.nan,
                **{c: np.asarray(record.get('acc_' + c) or [], float) for c in ('h1', 'h2', 'v')},
            )
        print(f'{min(start + 10, len(todo))} of {len(todo)}', flush=True)


if __name__ == '__main__':
    main()

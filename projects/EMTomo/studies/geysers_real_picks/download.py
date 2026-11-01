"""Download real EGS P picks for the existing 3776 Geysers events, all stations.

python -B studies/geysers_real_picks/download.py [--offline]
Caches original official Hypoinverse responses; never changes synthetic inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'data/geysers_2011/real_picks'
SOURCE_ID = 'geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_noise01_pick001'
SOURCE = ROOT.parent / 'forward_modeling/experiments/input' / SOURCE_ID
URL = 'https://ncedc.org/cgi-bin/catalog-search2.pl'
COLUMNS = ('event_id', 'origin_time_utc', 'network', 'station', 'location', 'channel',
           'phase', 'onset', 'polarity', 'weight_code', 'pick_time_utc',
           'seconds_from_catalog_origin', 'seconds_from_first_P',
           'in_existing_26_station_geometry', 'source_file', 'source_line')


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def iso(value):
    return value.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows, columns):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_document(text, wanted_ids, filename):
    """Read fixed-width Hypoinverse archive columns, preserving every P record.

    Origin ID: cols 137-146. Station/network/channel: 1-5/6-7/10-12.
    P label/onset: 14-15; polarity: 16; weight: 17; minute: 18-29;
    P seconds: 30-34 (implied two decimals); location: 112-113.
    Weight codes are NOT converted into uncertainties or used to reject picks.
    """
    match = re.search(r'<pre\b[^>]*>(.*?)</pre>', text, re.I | re.S)
    if match is None:
        raise ValueError(f'{filename}: missing complete PRE block')
    picks, origins = [], {}
    current, origin = None, None
    for number, line in enumerate(html.unescape(match[1]).splitlines(), 1):
        if not line.strip() or line.startswith('$'):
            continue
        if re.match(r'^\d{14}', line):
            if current is not None:
                raise ValueError(f'{filename}:{number}: previous event not terminated')
            current = line[136:146].strip()
            if not current.isdigit():
                raise ValueError(f'{filename}:{number}: invalid origin ID')
            origin = datetime.strptime(line[:12], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            origin += timedelta(seconds=float(Decimal(line[12:16].strip()) / 100))
            if current in wanted_ids:
                if current in origins and origins[current] != origin:
                    raise ValueError(f'{filename}: conflicting origins for {current}')
                origins[current] = origin
        elif line.strip().isdigit():
            if current is None or line.strip() != current:
                raise ValueError(f'{filename}:{number}: event terminator mismatch')
            current = None
        elif current is None:
            raise ValueError(f'{filename}:{number}: unexpected non-event line: {line!r}')
        elif current in wanted_ids and line[13:15].strip().endswith('P'):
            minute = datetime.strptime(line[17:29], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            seconds = line[29:34].strip()
            seconds = Decimal(seconds) if '.' in seconds else Decimal(seconds) / 100
            picked = minute + timedelta(seconds=float(seconds))
            station, network, channel = line[:5].strip(), line[5:7].strip(), line[9:12].strip()
            if not station or not network or not channel:
                raise ValueError(f'{filename}:{number}: incomplete P station identifier')
            picks.append(dict(event_id=current, origin_time_utc=iso(origin), network=network,
                              station=station, location=line[111:113].strip().replace('--', ''),
                              channel=channel, phase='P', onset=line[13:14].strip(),
                              polarity=line[15:16].strip(), weight_code=line[16:17].strip(),
                              pick_time_utc=iso(picked), seconds_from_catalog_origin=(picked-origin).total_seconds(),
                              source_file=filename, source_line=number))
    if current is not None:
        raise ValueError(f'{filename}: incomplete final event')
    return picks, origins


def parameters(day):
    return dict(catalog='BG', format='egsphraw', mintime=day.strftime('%Y/%m/%d,00:00:00'),
                maxtime=(day + timedelta(days=1)).strftime('%Y/%m/%d,00:00:00'),
                minmag='-2', maxmag='10', outputloc='web', searchlimit='100000')


def download_day(day, wanted, offline):
    path = OUTPUT / 'raw' / f'{day:%Y-%m-%d}.html'
    params = parameters(day)
    for attempt in range(4):
        try:
            if path.exists():
                data = path.read_bytes()
            elif offline:
                raise FileNotFoundError(path)
            else:
                request = urllib.request.Request(URL, data=urllib.parse.urlencode(params).encode(),
                                                 headers={'User-Agent': 'EMTomo-Geysers-research/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
            text = data.decode('utf-8')
            if f'start_time={params["mintime"]}' not in text or 'format=egsphraw' not in text or 'catalog=BG' not in text:
                raise ValueError(f'{path.name}: cached response query mismatch')
            picks, origins = parse_document(text, wanted, path.name)
            if not path.exists():
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(data)
                temporary.replace(path)
            return picks, origins, dict(file=str(path.relative_to(OUTPUT)), sha256=sha(path),
                                         bytes=len(data), endpoint=URL, method='POST', parameters=params,
                                         selected_events=len(origins), selected_P_records=len(picks))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if offline or attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def main(offline=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'raw').mkdir(exist_ok=True)
    expected_hashes = {'events.csv': 'd7f6af385a08edf1b4f622e35c08d6f40bbc06d2d60e3ab9e42a730e3dedd786',
                       'stations.csv': '59895d722cff63e2f087438704631ae6740f2f76dd7065ce64bad8e6b69be4e3'}
    if any(sha(SOURCE / name) != value for name, value in expected_hashes.items()):
        raise ValueError('Source geometry differs from approved 3776-event Geysers selection')
    with (SOURCE / 'events.csv').open() as stream:
        ids = [row['event_id'] for row in csv.DictReader(stream)]
    with (SOURCE / 'stations.csv').open() as stream:
        geometry = {('BG', row['station_id']) for row in csv.DictReader(stream)}
    assert len(ids) == len(set(ids)) == 3776 and len(geometry) == 26
    source_generation = json.loads((SOURCE / 'generation.json').read_text())
    expected_origins = {event: utc(source_generation['event_origin_times_utc'][event]) for event in ids}
    days = sorted({value.date() for value in expected_origins.values()})
    state = dict(status='running', requested_events=len(ids), completed_days=0,
                 total_days=len(days), updated_utc=iso(datetime.now(timezone.utc)))
    atomic_json(OUTPUT / 'status.json', state)
    picks, origins, requests = [], {}, []
    try:
        for day in days:
            rows, found, ledger = download_day(day, set(ids), offline)
            for event, origin in found.items():
                if event in origins and origins[event] != origin:
                    raise ValueError(f'Conflicting origins across responses for {event}')
                origins[event] = origin
            picks.extend(rows)
            requests.append(ledger)
            state.update(completed_days=len(requests), downloaded_P_records=len(picks),
                         matched_events=len(origins), updated_utc=iso(datetime.now(timezone.utc)))
            atomic_json(OUTPUT / 'status.json', state)
            print(f'{day}: {len(found)} selected events, {len(rows)} P records; '
                  f'total {len(origins)}/{len(ids)} events', flush=True)
            if not offline:
                time.sleep(.3)
        unique, duplicates = {}, 0
        for row in picks:
            key = tuple(row[name] for name in COLUMNS if name not in
                        ('seconds_from_first_P', 'in_existing_26_station_geometry', 'source_file', 'source_line'))
            if key in unique:
                duplicates += 1
            else:
                unique[key] = row
        picks = sorted(unique.values(), key=lambda row: (expected_origins[row['event_id']], row['event_id'],
                                                        row['pick_time_utc'], row['network'], row['station'], row['channel']))
        event_rows, station_events = defaultdict(list), defaultdict(set)
        for row in picks:
            event_rows[row['event_id']].append(row)
            station_events[(row['network'], row['station'])].add(row['event_id'])
        for rows in event_rows.values():
            earliest = min(utc(row['pick_time_utc']) for row in rows)
            for row in rows:
                row['seconds_from_first_P'] = (utc(row['pick_time_utc'])-earliest).total_seconds()
                row['in_existing_26_station_geometry'] = (row['network'], row['station']) in geometry
        event_coverage = [dict(event_id=event, in_downloaded_catalog=event in origins,
                               P_picks=len(event_rows[event]),
                               stations=len({(r['network'], r['station']) for r in event_rows[event]}),
                               P_picks_existing26=sum(r['in_existing_26_station_geometry'] for r in event_rows[event]))
                          for event in ids]
        station_counts = Counter((r['network'], r['station']) for r in picks)
        station_coverage = [dict(network=net, station=sta, P_picks=station_counts[(net, sta)],
                                 events=len(events), in_existing_26_station_geometry=(net, sta) in geometry)
                            for (net, sta), events in sorted(station_events.items())]
        write_csv(OUTPUT / 'P_picks.csv', picks, COLUMNS)
        write_csv(OUTPUT / 'event_coverage.csv', event_coverage, event_coverage[0].keys())
        write_csv(OUTPUT / 'station_coverage.csv', station_coverage,
                  ('network', 'station', 'P_picks', 'events', 'in_existing_26_station_geometry'))
        pair_counts = Counter((r['event_id'], r['network'], r['station']) for r in picks)
        shifts = {event: (origin-expected_origins[event]).total_seconds() for event, origin in origins.items()}
        summary = dict(source_experiment_id=SOURCE_ID, source_input_sha256=expected_hashes,
                       requested_events=len(ids), catalog_events_found=len(origins), events_with_P=len([v for v in event_rows.values() if v]),
                       missing_catalog_event_ids=[e for e in ids if e not in origins],
                       event_ids_without_P=[e for e in ids if not event_rows[e]],
                       P_picks=len(picks), stations=len(station_events),
                       P_picks_existing26=sum(r['in_existing_26_station_geometry'] for r in picks),
                       P_picks_other_stations=sum(not r['in_existing_26_station_geometry'] for r in picks),
                       unique_event_station_pairs=len(pair_counts), pairs_with_multiple_P_picks=sum(n > 1 for n in pair_counts.values()),
                       exact_duplicate_records_removed=duplicates,
                       weight_codes=dict(Counter(r['weight_code'] for r in picks)),
                       negative_seconds_from_catalog_origin=sum(r['seconds_from_catalog_origin'] < 0 for r in picks),
                       catalog_origins_changed=sum(abs(s) > .011 for s in shifts.values()),
                       max_catalog_origin_shift_s=max(map(abs, shifts.values()), default=0),
                       filtering='Exact existing event IDs; all stations/channels; P records only; no weight/quality rejection.',
                       station_scope='All reporting stations in the EGS/BG event phase archive, not limited to our 26.',
                       timing='Observed UTC picks; catalog origin is not ground truth. Relative times use earliest downloaded P per event.',
                       weight_policy='Preserved raw Hypoinverse weight codes; not interpreted as timing uncertainty.',
                       requests=requests, created_utc=iso(datetime.now(timezone.utc)))
        atomic_json(OUTPUT / 'summary.json', summary)
        state.update(status='completed', P_picks=len(picks), stations=len(station_events),
                     events_with_P=summary['events_with_P'], updated_utc=iso(datetime.now(timezone.utc)))
        atomic_json(OUTPUT / 'status.json', state)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('requests', 'event_ids_without_P', 'missing_catalog_event_ids')}, indent=2))
    except BaseException as error:
        state.update(status='failed', error=repr(error), updated_utc=iso(datetime.now(timezone.utc)))
        atomic_json(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Rebuild tables using cached raw responses only')
    main(parser.parse_args().offline)

"""Offline, solver-free contract tests for the Hypoinverse document parser."""

from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest


# Load only the downloader: no application/solver imports or main() side effects.
_spec = importlib.util.spec_from_file_location(
    "geysers_real_picks_download", Path(__file__).with_name("download.py")
)
_download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_download)
parse_document = _download.parse_document

FILENAME = "2011-04-01.html"
EVENT_ID = "3156099"
ORIGIN = datetime(2011, 4, 1, 0, 3, 12, 130000, tzinfo=timezone.utc)

# Verbatim first event from data/geysers_2011/real_picks/raw/2011-04-01.html.
# Keep column spacing; the tests never read the raw archive at runtime.
FIRST_EVENT = (
    "201104010003121338 4974122 4828  158     0  0  0   8  0 0   0  0 0   0         0    0  19  32  0                        0                  3156099L 75   0        1F    G",
    "ACR  BG  DPZ  P 5201104010003 1329   0  0        0                   0       0  0            0         0    B  --       ",
    "AL3  BG  DPZ  P 5201104010003 1357   0  0        0                   0       0  0            0         0    B  --       ",
    "AL4  BG  DPN    4201104010003             1360 S 7   0           0       0   0  0            0             0B  --       ",
    "AL4  BG  DPZ  P 7201104010003 1293   0  0        0                   0       0  0            0         0    B  --       ",
    "AL5  BG  DPZ  P 7201104010003 1348   0  0        0                   0       0  0            0         0    B  --       ",
    "AL6  BG  DPZ  P 7201104010003 1352   0  0        0                   0       0  0            0         0    B  --       ",
    "BRP  BG  DPE    4201104010003             1369 S 7   0           0       0   0  0            0             0B  --       ",
    "BRP  BG  DPZ  P 5201104010003 1301   0  0        0                   0       0  0            0         0    B  --       ",
    "BUC  BG  DPZ  P 5201104010003 1296   0  0        0                   0       0  0            0         0    B  --       ",
    "CLV  BG  DPE    4201104010003             1352 S 7   0           0       0   0  0            0             0B  --       ",
    "CLV  BG  DPZ  P 5201104010003 1284   0  0        0                   0       0  0            0         0    B  --       ",
    "HVC  BG  DPZ  P 7201104010003 1301   0  0        0                   0       0  0            0         0    B  --       ",
    "MCL  BG  DPZ  P 5201104010003 1309   0  0        0                   0       0  0            0         0    B  --       ",
    "NEG  BG  DPZ  P 7201104010003 1310   0  0        0                   0       0  0            0         0    B  --       ",
    "RGP  BG  DPZ  P 7201104010003 1347   0  0        0                   0       0  0            0         0    B  --       ",
    "SB4  BG  DPE    4201104010003             1349 S 7   0           0       0   0  0            0             0B  --       ",
    "SB4  BG  DPZ  P 7201104010003 1292   0  0        0                   0       0  0            0         0    B  --       ",
    "SQK  BG  DPZ  P 5201104010003 1265   0  0        0                   0       0  0            0         0    B  --       ",
    "                                                                 3156099",
)
HEADER, TERMINATOR = FIRST_EVENT[0], FIRST_EVENT[-1]
RGP_P, SB4_P = FIRST_EVENT[15], FIRST_EVENT[17]


def document(lines):
    return "<html><PRE>\n" + "\n".join(lines) + "\n</PRE></html>"


def replace_columns(line, start, stop, value):
    assert len(value) == stop - start
    return line[:start] + value + line[stop:]


def test_first_real_event_has_14_p_picks_and_excludes_4_s_records():
    picks, origins = parse_document(document(FIRST_EVENT), {EVENT_ID}, FILENAME)

    assert origins == {EVENT_ID: ORIGIN}
    assert origins[EVENT_ID].tzinfo is timezone.utc
    assert len(picks) == 14
    assert {p["station"] for p in picks} == {
        "ACR", "AL3", "AL4", "AL5", "AL6", "BRP", "BUC", "CLV",
        "HVC", "MCL", "NEG", "RGP", "SB4", "SQK",
    }
    assert {p["phase"] for p in picks} == {"P"}
    assert {p["channel"] for p in picks} == {"DPZ"}
    assert {p["event_id"] for p in picks} == {EVENT_ID}
    assert {p["origin_time_utc"] for p in picks} == {"2011-04-01T00:03:12.130Z"}
    assert {p["network"] for p in picks} == {"BG"}
    assert {p["location"] for p in picks} == {""}
    assert [p["source_line"] for p in picks] == [
        3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20,
    ]
    assert {p["source_file"] for p in picks} == {FILENAME}
    assert [p["weight_code"] for p in picks] == [
        "5", "5", "7", "7", "7", "5", "5", "5", "7", "5", "7", "7", "7", "5",
    ]


@pytest.mark.parametrize("station, timestamp, relative_seconds, source_line", [
    ("RGP", "2011-04-01T00:03:13.470Z", 1.34, 17),
    ("SB4", "2011-04-01T00:03:12.920Z", 0.79, 19),
])
def test_known_p_arrivals(station, timestamp, relative_seconds, source_line):
    picks, _ = parse_document(document(FIRST_EVENT), {EVENT_ID}, FILENAME)
    pick, = [p for p in picks if p["station"] == station]

    assert pick == dict(
        event_id=EVENT_ID, origin_time_utc="2011-04-01T00:03:12.130Z",
        network="BG", station=station, location="", channel="DPZ", phase="P",
        onset="", polarity="", weight_code="7", pick_time_utc=timestamp,
        seconds_from_catalog_origin=pytest.approx(relative_seconds),
        source_file=FILENAME, source_line=source_line,
    )


@pytest.mark.parametrize("wanted, expected_ids", [
    ({EVENT_ID}, [EVENT_ID, EVENT_ID]),
    ({"3156100"}, ["3156100"]),
    ({EVENT_ID, "3156100"}, [EVENT_ID, EVENT_ID, "3156100"]),
    ({"315609"}, []),
    (set(), []),
])
def test_filters_picks_and_origins_by_exact_header_id(wanted, expected_ids):
    # A second synthetic event differs only in its fixed-width header/terminator ID.
    other_header = replace_columns(HEADER, 136, 146, "   3156100")
    lines = (HEADER, RGP_P, SB4_P, TERMINATOR,
             other_header, RGP_P, "                                                                 3156100")
    picks, origins = parse_document(document(lines), wanted, FILENAME)

    assert [p["event_id"] for p in picks] == expected_ids
    assert origins == {event_id: ORIGIN for event_id in expected_ids}


def test_header_id_comes_from_columns_137_through_146():
    header = replace_columns(HEADER, 120, 130, "   9999999")
    picks, origins = parse_document(
        document((header, RGP_P, TERMINATOR)), {EVENT_ID, "9999999"}, FILENAME
    )

    assert origins == {EVENT_ID: ORIGIN}
    assert [p["event_id"] for p in picks] == [EVENT_ID]


@pytest.mark.parametrize("bad_id", ["          ", "   BAD-ID!"])
def test_rejects_invalid_header_id(bad_id):
    header = replace_columns(HEADER, 136, 146, bad_id)
    with pytest.raises(ValueError, match="invalid origin ID") as error:
        parse_document(document((header, RGP_P, TERMINATOR)), {EVENT_ID}, FILENAME)
    assert FILENAME in str(error.value)


@pytest.mark.parametrize("text", [
    "<html>No archive here</html>",
    "<PRE>\n" + "\n".join(FIRST_EVENT),
    "\n".join(FIRST_EVENT) + "</PRE>",
])
def test_rejects_missing_or_incomplete_pre(text):
    with pytest.raises(ValueError, match="missing complete PRE block") as error:
        parse_document(text, {EVENT_ID}, FILENAME)
    assert FILENAME in str(error.value)


@pytest.mark.parametrize("lines, message", [
    ((HEADER, RGP_P), "incomplete final event"),
    ((HEADER, RGP_P, HEADER), "previous event not terminated"),
    ((HEADER, RGP_P, "                                                                 3156100"),
     "event terminator mismatch"),
])
def test_rejects_truncated_events_and_mismatched_terminators(lines, message):
    with pytest.raises(ValueError, match=message) as error:
        parse_document(document(lines), {EVENT_ID}, FILENAME)
    assert FILENAME in str(error.value)


@pytest.mark.parametrize("seconds_field, expected", [
    (" 6000", "2011-04-02T00:00:00.000Z"),
    (" 6123", "2011-04-02T00:00:01.230Z"),
    ("61.23", "2011-04-02T00:00:01.230Z"),
])
def test_pick_seconds_roll_over_minute_and_day(seconds_field, expected):
    line = replace_columns(RGP_P, 17, 29, "201104012359")
    line = replace_columns(line, 29, 34, seconds_field)
    picks, origins = parse_document(
        document((HEADER, line, TERMINATOR)), {EVENT_ID}, FILENAME
    )

    pick, = picks
    assert pick["pick_time_utc"] == expected
    assert pick["seconds_from_catalog_origin"] == pytest.approx(
        (datetime.fromisoformat(expected.replace("Z", "+00:00")) - ORIGIN).total_seconds()
    )
    assert origins == {EVENT_ID: ORIGIN}


@pytest.mark.parametrize("location, expected_location", [("--", ""), ("  ", ""), ("01", "01")])
def test_fixed_width_identifiers_onset_polarity_and_location(location, expected_location):
    line = replace_columns(RGP_P, 0, 5, "ABCDE")
    line = replace_columns(line, 5, 7, "XY")
    line = replace_columns(line, 9, 12, "HHZ")
    line = replace_columns(line, 13, 15, "IP")
    line = replace_columns(line, 15, 16, "U")
    line = replace_columns(line, 111, 113, location)
    picks, _ = parse_document(document((HEADER, line, TERMINATOR)), {EVENT_ID}, FILENAME)

    pick, = picks
    assert (pick["station"], pick["network"], pick["channel"]) == ("ABCDE", "XY", "HHZ")
    assert (pick["phase"], pick["onset"], pick["polarity"], pick["weight_code"]) == (
        "P", "I", "U", "7",
    )
    assert pick["location"] == expected_location

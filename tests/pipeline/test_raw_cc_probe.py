from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("warcio")
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from web_curation_lab.pipeline.stages.raw_cc.probe import load_probe_config, run_probe


def _write_response(
    writer: WARCWriter,
    *,
    url: str,
    body: bytes,
    content_type: str,
    identified_mime: str,
) -> None:
    http_headers = StatusAndHeaders(
        "200 OK",
        [("Content-Type", content_type)],
        protocol="HTTP/1.1",
    )
    record = writer.create_warc_record(
        url,
        "response",
        payload=BytesIO(body),
        http_headers=http_headers,
        warc_headers_dict={"WARC-Identified-Payload-Type": identified_mime},
    )
    writer.write_record(record)


def _make_warc(path: Path) -> None:
    with path.open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        writer.write_record(writer.create_warcinfo_record("fixture.warc.gz", {"software": "test"}))
        _write_response(
            writer,
            url="https://example.com/",
            body=b"<html><body>Hello</body></html>",
            content_type="text/html; charset=utf-8",
            identified_mime="text/html",
        )
        _write_response(
            writer,
            url="https://example.com/data.json",
            body=b'{"answer": 42}',
            content_type="application/json",
            identified_mime="application/json",
        )
        request_headers = StatusAndHeaders(
            "GET /data.json HTTP/1.1",
            [("Host", "example.com")],
            protocol="",
        )
        writer.write_record(
            writer.create_warc_record(
                "https://example.com/data.json",
                "request",
                payload=BytesIO(),
                http_headers=request_headers,
            )
        )


def _write_config(path: Path, warc_path: Path, output_dir: Path, limit: int = 4) -> None:
    path.write_text(
        "\n".join(
            [
                "[stage]",
                'id = "0_raw_cc"',
                'crawl_id = "CC-MAIN-TEST"',
                "",
                "[probe]",
                f'input_path = "{warc_path}"',
                f'output_dir = "{output_dir}"',
                f"limit = {limit}",
                "preview_chars = 12",
                "",
            ]
        )
    )


def test_probe_preserves_raw_records_and_reports_datatrove_drops(tmp_path: Path) -> None:
    warc_path = tmp_path / "fixture.warc.gz"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "probe.toml"
    _make_warc(warc_path)
    _write_config(config_path, warc_path, output_dir)

    summary = run_probe(load_probe_config(config_path))
    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]

    assert summary["probe"]["records_inspected"] == 4
    assert summary["counts"]["record_types"] == {
        "request": 1,
        "response": 2,
        "warcinfo": 1,
    }
    assert summary["counts"]["datatrove_standard_reader_decisions"] == {
        "kept": 1,
        "mime_type": 1,
        "record_type": 2,
    }
    assert records[1]["preview"] == "<html><body>"
    assert records[1]["datatrove_standard_reader_would_keep"] is True
    assert records[2]["http_mime_type"] == "application/json"
    assert records[2]["datatrove_standard_reader_would_keep"] is False
    assert records[3]["record_type"] == "request"
    assert records[3]["http_status"] is None
    assert json.loads((output_dir / "summary.json").read_text())["input"]["sha256"]


def test_limit_override_counts_raw_warc_records(tmp_path: Path) -> None:
    warc_path = tmp_path / "fixture.warc.gz"
    output_dir = tmp_path / "output"
    config_path = tmp_path / "probe.toml"
    _make_warc(warc_path)
    _write_config(config_path, warc_path, output_dir)

    config = load_probe_config(config_path, limit_override=2)
    summary = run_probe(config)

    assert summary["probe"]["requested_record_limit"] == 2
    assert summary["probe"]["records_inspected"] == 2


def test_wrong_stage_id_fails_loudly(tmp_path: Path) -> None:
    config_path = tmp_path / "probe.toml"
    config_path.write_text(
        """
[stage]
id = "raw"
crawl_id = "CC-MAIN-TEST"

[probe]
input_path = "input.warc.gz"
output_dir = "output"
limit = 1
preview_chars = 1
""".strip()
    )

    with pytest.raises(ValueError, match="Expected stage.id"):
        load_probe_config(config_path)

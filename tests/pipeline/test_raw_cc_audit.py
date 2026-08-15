from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("warcio")
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

import web_curation_lab.pipeline.stages.raw_cc.audit as audit_module
import web_curation_lab.pipeline.stages.raw_cc.reader as reader_module
from web_curation_lab.pipeline.stages.raw_cc.audit import load_audit_config, run_audit


def _write_response(
    writer: WARCWriter,
    *,
    url: str,
    body: bytes,
    http_content_type: str,
    identified_mime: str,
) -> None:
    record = writer.create_warc_record(
        url,
        "response",
        payload=BytesIO(body),
        http_headers=StatusAndHeaders(
            "200 OK",
            [("Content-Type", http_content_type)],
            protocol="HTTP/1.1",
        ),
        warc_headers_dict={"WARC-Identified-Payload-Type": identified_mime},
    )
    writer.write_record(record)


def _write_fixture(path: Path) -> None:
    with path.open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        _write_response(
            writer,
            url="https://example.com/disagreement",
            body=b"visible text",
            http_content_type="application/json; charset=utf-8",
            identified_mime="text/html",
        )
        _write_response(
            writer,
            url="https://example.com/image",
            body=b"PNG payload",
            http_content_type="image/png",
            identified_mime="image/png",
        )
        _write_response(
            writer,
            url="https://example.com/decode",
            body=b"\xff",
            http_content_type="text/plain; charset=utf-8",
            identified_mime="text/plain",
        )
        _write_response(
            writer,
            url="https://example.com/empty",
            body=b"",
            http_content_type="text/plain; charset=utf-8",
            identified_mime="text/plain",
        )


def _write_config(path: Path, warc_path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
[stage]
id = "0_raw_cc"
crawl_id = "CC-MAIN-TEST"
policy_revision = "0_raw_cc-test-v1"

[probe]
input_path = "{warc_path}"

[census]
text_mime_prefixes = ["text/"]
text_mime_types = ["application/json"]
text_mime_suffixes = ["+json", "+xml"]

[audit]
output_dir = "{output_dir}"
samples_per_category = 2
preview_bytes = 4
preview_chars = 5
""".strip()
    )


def test_audit_uses_reader_policy_and_writes_bounded_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warc_path = tmp_path / "fixture.warc.gz"
    output_dir = tmp_path / "audit"
    config_path = tmp_path / "config.toml"
    _write_fixture(warc_path)
    _write_config(config_path, warc_path, output_dir)

    def fake_detect(payload: bytes, magic_module: object) -> tuple[str | None, None, float]:
        del magic_module
        if payload.startswith(b"PNG"):
            return "image/png", None, 1.0
        if not payload:
            return None, None, 1.0
        return "text/plain", None, 1.0

    monkeypatch.setattr(reader_module, "_detect_payload", fake_detect)
    monkeypatch.setattr(audit_module, "_load_magic", lambda: (object(), None))

    summary = run_audit(load_audit_config(config_path))
    rows = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]

    assert summary["counts"]["reader_outcomes"] == {
        "accepted": 1,
        "accepted_with_replacement": 1,
        "dropped_empty_text": 1,
        "dropped_non_text_mime": 1,
    }
    assert summary["counts"]["audit_categories"] == {
        "non_text_mime": 1,
        "replacement_decode": 1,
        "empty_text": 1,
        "mime_disagreement": 1,
    }
    assert summary["counts"]["audit_category_outcomes"]["mime_disagreement"] == {
        "accepted": 1
    }
    assert summary["counts"]["decode_failure_replacement_audit"][
        "priority_candidate_replacement_ratio_buckets"
    ] == {"gt_1_percent": 1}
    assert {row["audit_category"] for row in rows} == {
        "non_text_mime",
        "replacement_decode",
        "empty_text",
        "mime_disagreement",
    }
    image = next(row for row in rows if row["audit_category"] == "non_text_mime")
    assert image["payload_prefix_bytes"] == 4
    assert image["payload_prefix_truncated"] is True
    disagreement = next(row for row in rows if row["audit_category"] == "mime_disagreement")
    assert disagreement["text_preview"] == "visib"
    assert disagreement["reader_outcome"] == "accepted"
    decode_failure = next(
        row for row in rows if row["audit_category"] == "replacement_decode"
    )
    assert decode_failure["reader_outcome"] == "accepted_with_replacement"
    assert decode_failure["replacement_characters"] == 1
    assert decode_failure["replacement_decode_options"] == [
        {
            "decoded_characters": 1,
            "encoding": "utf-8",
            "replacement_characters": 1,
            "replacement_ratio": 1.0,
        }
    ]


def test_audit_config_rejects_unbounded_sample_size(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[stage]
id = "0_raw_cc"
crawl_id = "CC-MAIN-TEST"
policy_revision = "0_raw_cc-test-v1"
[probe]
input_path = "fixture.warc.gz"
[census]
text_mime_prefixes = ["text/"]
text_mime_types = []
text_mime_suffixes = []
[audit]
output_dir = "audit"
samples_per_category = 0
preview_bytes = 1
preview_chars = 1
""".strip()
    )

    with pytest.raises(ValueError, match="samples_per_category must be positive"):
        load_audit_config(config_path)

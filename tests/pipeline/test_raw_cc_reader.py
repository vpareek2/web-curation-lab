from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("warcio")
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from web_curation_lab.pipeline.stages.raw_cc.reader import RawWarcReader


def _write_response(
    writer: WARCWriter,
    *,
    url: str,
    body: bytes,
    mime_type: str,
) -> None:
    record = writer.create_warc_record(
        url,
        "response",
        payload=BytesIO(body),
        http_headers=StatusAndHeaders(
            "200 OK",
            [("Content-Type", f"{mime_type}; charset=utf-8")],
            protocol="HTTP/1.1",
        ),
        warc_headers_dict={"WARC-Identified-Payload-Type": mime_type},
    )
    writer.write_record(record)


def test_raw_reader_keeps_configured_text_without_extracting_html(tmp_path: Path) -> None:
    warc_path = tmp_path / "fixture.warc.gz"
    with warc_path.open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        writer.write_record(writer.create_warcinfo_record("fixture.warc.gz", {"software": "test"}))
        _write_response(
            writer,
            url="https://example.com/",
            body=b"<html><nav>raw navigation</nav></html>",
            mime_type="text/html",
        )
        _write_response(
            writer,
            url="https://example.com/data.json",
            body=b'{"answer": 42}',
            mime_type="application/json",
        )
        _write_response(
            writer,
            url="https://example.com/image.png",
            body=b"\x89PNG\r\n\x1a\n",
            mime_type="image/png",
        )
        _write_response(
            writer,
            url="https://example.com/broken.txt",
            body=b"mostly valid utf-8 \xff text",
            mime_type="text/plain",
        )

    reader = RawWarcReader(
        str(tmp_path),
        crawl_id="CC-MAIN-TEST",
        policy_revision="0_raw_cc-test-v1",
        glob_pattern=warc_path.name,
        text_mime_prefixes=("text/",),
        text_mime_types=("application/json",),
        text_mime_suffixes=("+json", "+xml"),
    )
    deepcopy(reader)
    documents = list(reader())

    assert [document.text for document in documents] == [
        "<html><nav>raw navigation</nav></html>",
        '{"answer": 42}',
        "mostly valid utf-8 \ufffd text",
    ]
    assert documents[0].metadata["stage_id"] == "0_raw_cc"
    assert documents[0].metadata["policy_revision"] == "0_raw_cc-test-v1"
    assert documents[0].metadata["effective_mime_type"] == "text/html"
    assert documents[1].metadata["effective_mime_type"] == "application/json"
    assert documents[2].metadata["decode_used_replacement"] is True
    assert documents[2].metadata["replacement_characters"] == 1
    assert reader.stats["response_records"].total == 4
    assert reader.stats["decoded_documents"].total == 3
    assert reader.stats["dropped_non_text_mime"].total == 1
    assert reader.stats["decoded_with_replacement"].total == 1
    assert reader.stats["replacement_characters"].total == 1

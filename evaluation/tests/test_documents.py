from web_curation_eval.documents import infer_domain, segment_document


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(10, 10 + len(text)))


def test_document_segments_are_disjoint_and_count_bytes_once() -> None:
    segments = list(
        segment_document(
            source="source",
            record={"text": "abcde", "domain": "example"},
            tokenizer=FakeTokenizer(),
            max_sequence_length=4,
        )
    )

    assert [segment.token_ids for segment in segments] == [[1, 10, 11, 12], [1, 13, 14]]
    assert sum(segment.document_bytes for segment in segments) == 5
    assert sum(segment.document_count for segment in segments) == 1
    assert all(segment.domain == "source/example" for segment in segments)


def test_nested_domain_and_empty_text() -> None:
    assert infer_domain("src", {"metadata": {"subreddit": "python"}}) == "src/python"
    assert list(
        segment_document(
            source="src",
            record={"text": ""},
            tokenizer=FakeTokenizer(),
            max_sequence_length=4,
        )
    ) == []

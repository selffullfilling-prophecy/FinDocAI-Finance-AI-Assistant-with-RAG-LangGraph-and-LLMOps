from frontend.streamlit_app import split_stream_chunk


def test_split_stream_chunk_returns_multiple_word_pieces():
    pieces = split_stream_chunk("Apple's total net sales decreased")

    assert len(pieces) > 1
    assert "".join(pieces) == "Apple's total net sales decreased"
    assert pieces[0] == "Apple's "


def test_split_stream_chunk_splits_long_unspaced_text():
    text = "A" * 40

    pieces = split_stream_chunk(text)

    assert len(pieces) > 1
    assert "".join(pieces) == text


def test_split_stream_chunk_empty_text_returns_empty_list():
    assert split_stream_chunk("") == []

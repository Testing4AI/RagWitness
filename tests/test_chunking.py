from ragmetatest.chunking import chunk_document
from ragmetatest.types import Document


def test_word_chunking_matches_expected_stride():
    document = Document(doc_id="d", version=1, text=" ".join(f"w{i}" for i in range(600)))
    chunks = chunk_document(document, chunk_size=256, chunk_overlap=20)
    assert [len(chunk.text.split()) for chunk in chunks] == [256, 256, 128]
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert chunks[1].text.split()[-20:] == chunks[2].text.split()[:20]


def test_chunk_identity_changes_with_version():
    v1 = Document(doc_id="d", version=1, text="same text")
    v2 = Document(doc_id="d", version=2, text="same text")
    assert chunk_document(v1)[0].chunk_id != chunk_document(v2)[0].chunk_id


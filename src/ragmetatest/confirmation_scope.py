"""Frozen confirmation units, independent of observed result-file contents."""

TRACE_FAULTS = {
    "update": ("clean", "dropped_update", "orphan_old_chunks", "partial_rechunk", "stale_embedding_vector", "metadata_vector_skew", "mixed_version_retrieval"),
    "warm_update": ("clean", "stale_query_cache"), "replay": ("clean", "non_idempotent_replay"),
    "rollback": ("clean", "rollback_leaves_new_chunks"), "delete": ("clean", "dropped_delete", "partial_delete"),
    "rename": ("clean", "rename_leaves_old_chunks", "rename_drops_new_document"),
    "out_of_order": ("clean", "out_of_order_update"),
}


def expected_answer_units(case_ids, models):
    return {(case, backend, top_k, trace, fault, model)
            for case in case_ids for backend in ("bm25", "chroma-contriever", "qdrant-contriever")
            for top_k in (1, 5) for trace, faults in TRACE_FAULTS.items()
            for fault in faults for model in models}


def answer_unit(row):
    return tuple(row[key] for key in ("case_id", "retriever", "top_k", "trace", "fault", "target_model"))

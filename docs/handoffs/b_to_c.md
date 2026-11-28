# Handoff from Member B to Member C

The frozen retrieval cache stores five ranked `chunks` per `qid` for Recall@5.
Use the first `retrieval.top_k_context` entries (four) to build RAG prompts.
Keep `section_ids` as strings such as `8.1` and `6.1.10`.

- The Week 2 runner reads the cache and passes all `chunks` directly to
  `RAGPipeline.run_query`. Add the top-four slice before constructing the
  prompt so live runs match the prompt budget measured on dev.
- A real RAG run with a missing cache or missing `qid` should stop with the
  missing path or ID. The existing sample passage can still serve the explicit
  mock mode. This protects experiment records from mixing sample and real inputs.

[Project spec §6](../project_spec.md#6-retrieval-run-once-then-frozen) | [Member C PR #8](https://github.com/motoric-o/academic-rag-chatbot/pull/8).

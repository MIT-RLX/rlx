# Dual-encoder retrieval for the KV context store — design plan

**Status:** plan (not implemented). Follow-on to Q-export (Q·K block scoring).
**Owner seam:** `rlx_runtime::kv_context_store` + `rlx-qwen3` generator wiring.

## Why (the ceiling that Q·K / K·K / lexical all hit)

Today a block's retrieval key is derived from the *generation model's own KV
space*:

- **K·K** (proxy): key = mean of the block's post-RoPE K rows; query = newest
  token's K. Ranks by key-self-similarity — a weak stand-in for attention.
- **Q·K** (the just-shipped root fix): query = the model's actual post-RoPE
  attention query (GQA-pooled to `kv_dim`). Ranks by the geometry the model
  *itself* uses to attend. Strictly better than K·K, but still bounded by two
  properties of the KV space:
  1. **Position entanglement.** K (and Q) are RoPE-rotated. A block's key is a
     mean of rotated vectors, so its "meaning" is smeared by *where the block
     sat*, not just *what it says*. Two paraphrases at different positions land
     far apart.
  2. **Objective mismatch.** K/V are optimized for next-token attention, not for
     "does this span contain the answer to this question." They are a decent but
     incidental retrieval signal.
- **Lexical (BM25-lite)** catches exact surface overlap (numbers, names) but
  misses paraphrase ("launch code" ↔ "access PIN").

None of these three is a *content* embedding. That's the gap a dual encoder
fills.

## What

Introduce a **separate, retrieval-optimized text encoder** (a second, small
model) that maps a text span → a dense vector in a space built for semantic
retrieval. Two towers over the *same* encoder (asymmetric via task prefix, as
nomic-embed/E5 do):

- **Document tower:** encode each offloaded block's decoded text
  (`search_document: …`) → store as the block's HNSW key.
- **Query tower:** encode the current question / recent-window text
  (`search_query: …`) → use as the HNSW query.

Retrieval becomes `embed(query) · embed(block_text)` — semantic, position-free.
Crucially, **only *selection* changes.** The retrieved *block ids* still splice
their original generation-model KV into the resident cache. The encoder decides
*which* KV to rehydrate; the KV used for attention is unchanged. So the encoder
is a pure retrieval contract, swappable and independently trainable.

## Concrete pieces already in the repo

- **Encoder:** `rlx-embed` → `RlxEmbed::from_dir_on(dir, Pooling::Mean, device)`
  with `dim()` and `embed_with_rlx(model, tok, texts, pooling)` (tokenize →
  forward → mean-pool → L2-normalize). nomic-embed-text is the natural default
  (768-d, retrieval-tuned, Matryoshka-truncatable to 256/128 for a smaller
  index). BERT-family also supported.
- **Store:** `KvContextStore` already keeps a per-block HNSW key + optional
  lexical tokens + provenance `Origin`. The HNSW is generic over vector dim.
- **Block text:** the generator already reconstructs a block's token span for
  lexical tagging (`self.tokens[start..start+rows]`); decode → string → encode.

## Seams / changes (model-agnostic core stays clean)

1. **Core (`rlx-runtime`), no model dep:** define a tiny trait
   ```rust
   pub trait BlockEmbedder: Send + Sync {
       fn embed_documents(&mut self, texts: &[&str]) -> Vec<Vec<f32>>;
       fn embed_query(&mut self, text: &str) -> Vec<f32>;
       fn dim(&self) -> usize;
   }
   ```
   Keep the concrete encoder out of core (mirrors
   [[feedback_rlx_framework_agnostic]]): core only sees the trait.

2. **`KvContextStore`:** allow the HNSW key space to be a *separate* embed dim
   (not `kv_dim`). Add:
   - `append_block_embed(start, origin, layer, k, v, embed_key)` — store the
     block's KV as today but key the HNSW on `embed_key` (len = `embedder.dim()`).
   - `retrieve_embed(embed_query, topk)` and an N-way `retrieve_hybrid3`
     blending **embed·embed + Q·K + lexical** with weights `(w_e, w_q, w_l)`.
   Keep the mean-K key stored alongside (cheap) so Q·K remains available as one
   of the blended signals.

3. **`rlx-qwen3` generator (can depend on `rlx-embed`), behind a
   `dual-encoder` feature** ([[feedback_optimizations_behind_features]],
   default OFF):
   - `KvStoreConfig::encoder(Arc<dyn BlockEmbedder>)` + blend weights.
   - Offload path: decode each evicted block's token span → text →
     `embed_documents` (batched) → `append_block_embed`.
   - Retrieval path: `search_query` = the recent-window text (≈ the question) →
     `embed_query` → `retrieve_hybrid3`.
   - Concrete `BlockEmbedder` = thin adapter over `RlxEmbed` + the tokenizer,
     living in rlx-qwen3 (or a shared `rlx-models-core` helper).

## Why it should beat Q·K (the hypotheses to bench)

- **Paraphrase recall:** plant "the launch code is 7731", ask "what's the access
  PIN?" — embeddings generalize where Q·K (token/position-tied) and BM25
  (surface-tied) both miss. This is the headline test: add *paraphrased* recall
  needles to `memory_probe`.
- **Position invariance:** a fact's retrievability no longer depends on where it
  landed in the stream.
- **Reusability:** the same index serves *any* generation model; the encoder is
  the contract. Aligns with the "reusable analysis/retrieval tools inside rlx"
  goal.
- **Independent tunability:** the encoder can be contrastively fine-tuned for the
  retrieval task without touching the LLM.

## Costs / risks

- **Second resident model + a forward per offloaded block** (append time) and
  **per query** (retrieval). nomic-embed is ~137M — cheap next to qwen3 decode,
  and block-embeds batch. Budget: amortize by embedding at block *seal* time,
  not per token.
- **Block granularity.** 16-token blocks may be too short for a meaningful
  embedding; consider embedding an overlapping *window* around the block, or
  raising block size for the embed key while keeping KV splice granularity.
- **Two spaces to maintain** (embed key for nav + mean-K for the Q·K blend).
- **Detokenization** for the document text (the generator has the tokenizer).

## Phased delivery

1. Core trait + `KvContextStore` embed-key path + `retrieve_embed` (+ unit test
   with a stub embedder: deterministic hashing encoder → asserts semantic-ish
   nearest-neighbor).
2. `retrieve_hybrid3` 3-way blend + weights on `KvStoreConfig`.
3. rlx-qwen3 `dual-encoder` feature: `RlxEmbed` adapter + offload/retrieve wiring.
4. `memory_probe`: `kvstore:…:enc[:W_E:W_Q:W_L]` spec field + **paraphrased**
   needles; bench recall vs Q·K and lexical. Record telemetry as usual.
5. If it wins: document in [[kv_retention_seam]] and `docs/kv-retention.md`.

## Relationship to shipped work

- Builds directly on Q-export (Q·K) — the dual encoder *adds* a semantic signal;
  it does not replace Q·K, which stays as one blended term.
- Orthogonal to the disk-tiering / HNSW / decay / provenance already shipped:
  those are the *index*; the dual encoder changes only the *key/query vectors*
  fed into it.

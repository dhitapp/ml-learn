# Grouped-query attention

Grouped-query attention (GQA) sits between multi-head attention (MHA) and
multi-query attention (MQA). The query heads stay separate, but several query
heads share the same key and value heads.

```text
G = n_heads       multi-head attention
1 < G < n_heads   grouped-query attention
G = 1             multi-query attention
```

Here, `G` is the number of key/value groups. For example, with four query heads
and two groups:

```text
query heads 0, 1  ->  K/V group 0
query heads 2, 3  ->  K/V group 1
```

The number of heads must therefore be divisible by the number of groups.

## Why share keys and values?

The main reason is the KV cache used during generation. When the model produces
one token at a time, it keeps the keys and values from previous tokens instead
of calculating them again. A simplified cache-size formula is:

```text
2 * n_layers * G * head_size * sequence_length * bytes_per_element
```

The `2` accounts for keys and values. More importantly, the cache depends on
the number of K/V groups, not the number of query heads. Reducing `G` therefore
reduces the cache while leaving the query heads alone.

MQA gives the smallest cache because all query heads use one K/V pair. GQA is a
compromise: it uses more than one K/V pair, which gives the model more capacity,
but still uses much less cache than ordinary MHA.

For a model with 64 query heads, the relative number of cached K/V heads is:

```text
MHA: G = 64   -> 64 K/V heads
GQA: G = 8    ->  8 K/V heads
MQA: G = 1    ->  1 K/V head
```

This does not remove the attention calculation itself. Every query head still
attends to the available token positions. GQA mainly reduces K/V parameters and
KV-cache memory during autoregressive generation.

## Implementation in this project

The implementation is in
[`grouped_query_attention.py`](grouped_query_attention.py). One linear layer
produces all grouped keys and another produces all grouped values:

```python
self.key = nn.Linear(embedding_size, n_groups * head_size)
self.value = nn.Linear(embedding_size, n_groups * head_size)
```

After projection, K and V are viewed as:

```text
(batch, time, n_groups, head_size)
```

Each query head chooses a group using integer division:

```python
group_size = n_heads // n_groups
group_index = head_index // group_size
```

The query projection still belongs to each individual `CausalAttention` head.
Only K and V are shared.

The layer also supports the KV cache. In evaluation mode, new keys and values
are appended along the time dimension and returned to the caller. Training does
not use the cache because the full sequence is processed at once.

## Parameter count

Let `d` be the embedding size, `h` the number of query heads, and `G` the number
of K/V groups. Since `head_size = d / h`, the attention parameters per block are:

```text
Q projections:   d^2
K projections:   d^2 * G/h
V projections:   d^2 * G/h
output project:  d^2 + d

total:           d^2 * (2 + 2G/h) + d
```

With this project's small configuration (`d = 64`, `h = 4`):

```text
              attention parameters per block
G = 4 (MHA)              16,448
G = 2 (GQA)              12,352
G = 1 (MQA)              10,304
```

The parameter difference is small at this scale. The more useful experiment is
to compare KV-cache size and generation speed as the context becomes longer.

## Converting an MHA checkpoint

The GQA paper also describes converting an existing MHA checkpoint. Within each
group, the K projection weights are averaged into one K projection, and the V
weights are averaged in the same way. The converted model is then trained for a
short additional period to recover quality. The paper calls this
**uptraining**.

That conversion is not implemented here; the current model trains its GQA
weights directly.

## Things to compare

Using the same model configuration and checkpoint settings, try `G = h`, an
intermediate value, and `G = 1`. Useful measurements are:

- parameter count;
- KV-cache memory;
- tokens generated per second;
- validation loss or generated-text quality.

The cache difference should become clearer with longer sequences. For very
short contexts, Python overhead and the small model size may hide most of the
speed difference.

## References

- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245)
- Shazeer, [*Fast Transformer Decoding: One Write-Head is All You
  Need*](https://arxiv.org/abs/1911.02150)
- Pope et al., [*Efficiently Scaling Transformer
  Inference*](https://arxiv.org/abs/2211.05102)

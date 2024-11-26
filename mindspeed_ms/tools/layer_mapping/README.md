TopLayer:
  network_with_loss: module

Embedding:
  dropout: embedding_dropout

ParallelTransformerLayer:
  attention: self_attention

ParallelAttention:
  qkv_proj: query_key_value
  out_proj: dense

ParallelMLP:
  mapping: dense_h_to_4h
  projection: dense_4h_to_h

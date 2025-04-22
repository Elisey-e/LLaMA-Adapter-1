
Currently, Image-Bind cannot be started due to an issue with CUDA drivers. Inference was only conducted for the LLama Adapter and other models.

## Setup

* Set up a new conda environment and install ImageBind with required packages:
  ```bash
  # Create virtual environment
  python3 -m venv .venv
  source .venv/bin/activate
  
  # Install ImageBind
  cd ImageBind
  pip install -r requirements.txt
  
  # Install additional dependencies
  cd ../
  pip install -r requirements.txt
  pip install -r req_fix.txt

  # Run demo
  python3 demo.py
  ```

## Rank-Adapter Token Processing

To better understand the authors' motivation for using a single visual token architecture rather than multiple tokens, let's examine the LLaMa-Adapter-v2 implementation.

**llama_adapter_v2_multimodal7b/llama/llama_adapter.py**

### Low-Rank Adapter Mechanism

Key architectural components:

```python
self.visual_query = nn.Embedding(query_len, v_embed_dim)
self.visual_blocks = nn.ModuleList([
    Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
    for _ in range(v_depth)])
self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
self.visual_proj_norm = nn.LayerNorm(model_args.dim)

# Adapter query
self.adapter_query = nn.Embedding(
    query_len * query_layer, model_args.dim)
```

Where `Block` refers to `timm.models.vision_transformer.Block` - the Visual Query Transformer.

Visual feature processing occurs in `forward_visual`:

```python
def forward_visual(self, imgs):
    clip_feats = self.clip_encode_image(imgs)
    clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

    visual_query = self.visual_query.weight.unsqueeze(
        0).repeat(len(imgs), 1, 1)
    visual_query = torch.cat([visual_query, clip_feats], dim=1)
    for block in self.visual_blocks:
        visual_query = block(visual_query)

    visual_query = visual_query[:, :self.query_len, :]
    visual_query = self.visual_proj(visual_query)
    visual_query = self.visual_proj_norm(visual_query)

    return visual_query
```

The `query_len` parameter controls the Image-Text adapter dimensionality, defaulting to the authors' recommended 10 tokens. These 10 tokens propagate visual feature information to the text model.

Key uses of `query_len`:

1. Defining trainable visual queries:
   ```python 
   self.visual_query = nn.Embedding(query_len, v_embed_dim)
   ```

2. Determining adapter query tensor size:
   ```python
   self.adapter_query = nn.Embedding(query_len * query_layer, model_args.dim)
   ```

3. Specifying the number of extracted tokens:
   ```python
   visual_query = visual_query[:, :self.query_len, :]
   ```

The visual features from `visual_query = self.forward_visual(imgs)` are then passed to the transformer model:

```python
adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
adapter_index = 0
for layer in self.llama.layers[-1 * self.query_layer:]:
    dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
    dynamic_adapter = dynamic_adapter + visual_query
    h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
    adapter_index = adapter_index + 1
```

### Examining Image-Bind

The implementation resides in `imagebind_LLM/ImageBind/models/imagebind_model.py`, with modality preprocessors defined in `_create_modality_preprocessors`.

Visual processing handled by:

```python
rgbt_preprocessor = RGBDTPreprocessor(
    img_size=[3, video_frames, 224, 224],
    num_cls_tokens=1,
    pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
    rgbt_stem=rgbt_stem,
    depth_stem=None,
)
```

Notable parameter `num_cls_tokens`:

```python
if self.num_cls_tokens > 0:
    self.cls_token = nn.Parameter(
        torch.zeros(1, self.num_cls_tokens, self.embed_dim)
    )

...

tokens = stem(input)
if self.num_cls_tokens > 0:
    class_tokens = self.cls_token.expand(
        B, -1, -1
    )  # class_tokens implementation from Phil Wang
    tokens = torch.cat((class_tokens, tokens), dim=1)
```

This parameter controls the number of tokens added to the embedding sequence for each modality (default: 1).

### Potential Improvements

While untested, modifying `num_cls_tokens` could potentially:
- Enhance visual data significance during training/inference
- Improve model performance on OCR, VQA benchmarks

However, challenges include:
- Impact on modality balance (authors optimized for equal weighting)
- Possible quality degradation for other modalities

Notably, ImageBind's performance is already close to LLaMa-Adapter, so gains might be marginal compared to cross-modal tradeoffs. Nevertheless, adjusting `num_cls_tokens` could help optimize models for specific tasks by rebalancing modality significance.

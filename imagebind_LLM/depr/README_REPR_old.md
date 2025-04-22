Currently, Image-Bind could not be started, there is a problem with the cuda drivers. The inference was conducted only for the LLama Adapter and other models.


## Setup

* Setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create venv
  python3 -m venv .venv
  source .venv/bin/activate
  # install ImageBind
  cd ImageBind
  pip install -r requirements.txt
  # install other dependencies
  cd ../
  pip install -r requirements.txt
  pip install -r req_fix.txt

  python3 demo.py
  ```

## Rank-Adapter processing tokens

Для того, чтобы лучше понять мотивацию авторов к построению архитектуры с одним визуальным токеном вместо большего числа, обратимся к реализации LLaMa-Adapter-v2.

**llama_adapter_v2_multimodal7b/llama/llama_adapter.py**

Механизм низкорангового адаптера 

Рассмотрим используемые архитектурные блоки

```
self.visual_query = nn.Embedding(query_len, v_embed_dim)
self.visual_blocks = nn.ModuleList([
    Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
    for _ in range(v_depth)])
self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
self.visual_proj_norm = nn.LayerNorm(model_args.dim)

# 3. adapter query
self.adapter_query = nn.Embedding(
    query_len * query_layer, model_args.dim)
```

где Block это timm.models.vision_transformer.Block - Visual Query Transformer

Обработкой визуальных признаков занимается функция forward_visual:

```
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

Параметр query_len отвечает за размерность Image-Text адаптера, по умолчанию установлен в указанные авторами 10 токенов. Эти 10 токенов и пропускают информацию о визуальных признаках в текстовую модель.

Собственно, ключевые места, где используется query_len это:

1)  Определение количества обучаемых визуальных запросов:
    self.visual_query = nn.Embedding(query_len, v_embed_dim)

2)  Определение размера тензора запросов адаптера:
    self.adapter_query = nn.Embedding(query_len * query_layer, model_args.dim)

3)  И определение количества токенов, которые извлекаются из запроса:
    visual_query = visual_query[:, :self.query_len, :]

Ну и визуальные признаки, полученные visual_query = self.forward_visual(imgs) далее передаются в саму модель-трансформер.

```
adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
adapter_index = 0
for layer in self.llama.layers[-1 * self.query_layer:]:
    dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
    dynamic_adapter = dynamic_adapter + visual_query
    h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
    adapter_index = adapter_index + 1
```

### Теперь обратимся к Image-Bind

imagebind_LLM/ImageBind/models/imagebind_model.py описывает реализацию модели, а в функции _create_modality_preprocessors описание обработчиков для всех модальностей.


За графику отвечает:

```
rgbt_preprocessor = RGBDTPreprocessor(
    img_size=[3, video_frames, 224, 224],
    num_cls_tokens=1,
    pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
    rgbt_stem=rgbt_stem,
    depth_stem=None,
)
```

В котором можно обратить внимание на параметр num_cls_tokens

```
if self.num_cls_tokens > 0:
  self.cls_token = nn.Parameter(
      torch.zeros(1, self.num_cls_tokens, self.embed_dim)
  )

...

tokens = stem(input)
if self.num_cls_tokens > 0:
  class_tokens = self.cls_token.expand(
      B, -1, -1
  )  # stole class_tokens impl from Phil Wang, thanks
  tokens = torch.cat((class_tokens, tokens), dim=1)
```

Который отвечает за число токенов для каждой модальности, которое добавляется к последовательности эмбеддингов.

По умолчанию он установлен в 1.

Изменяя этот параметр, можно добиться улучшение значимости визуальных данных в процессе обучения и инференса модели.

Проверить это возможности уже нет, но возможно, главная трудность - это влияние на относительную значимость модальностей, которые авторы настроили равноправными. Тем не менее, увеличением num_cls_tokens похоже, лействительно возможно добиться улучшения показателей модели на OCR, VQA и прочих бенчмарках, справедливости ради, следует отметить, что ImageBind несильно отстает по качеству от LLaMa-Adapter, так что это улучшение может быть не столь значимым по сравнению с падением качества для остальных модальностей.

Тем не менее, num_cls_tokens может позволить улучшить качество модели под конкретную задачу, сбалансировав значимость разных модальностей.


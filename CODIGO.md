# CODIGO.md — Anatomia do Código

Este documento percorre **cada arquivo** do projeto explicando o que ele faz, quais funções/classes contém, que shapes os tensores têm, e como tudo se conecta. É o companheiro de `RAFAEL.md` (que cobre a teoria) — aqui é tudo prático.

---

## Estrutura do Projeto

```
YOLO/
├── configs/
│   ├── vlm_config.yaml       # backend custom: CLIP + SmolLM
│   └── phi3_config.yaml      # backend phi3: Phi-3-vision-128k
├── data/
│   ├── download.py           # baixa dataset da URL no config
│   ├── label_to_caption.py   # bbox YOLO → frase em inglês
│   ├── yolo_dataset.py       # Dataset PyTorch para custom backend
│   └── phi3_dataset.py       # Dataset PyTorch para phi3 backend
├── models/
│   ├── vision_encoder.py     # wrapper CLIP frozen
│   ├── projection.py         # MLP vision_dim → language_dim
│   ├── language_decoder.py   # SmolLM + LoRA
│   ├── small_vlm.py          # custom: encoder + projection + decoder
│   └── phi3_vlm.py           # phi3: Phi-3-vision + LoRA + 4-bit opcional
├── train.py                  # loop de treino genérico (custom + phi3)
├── inference.py              # geração de captions
└── requirements.txt
```

---

## `configs/`

Os YAMLs definem **tudo o que muda entre runs**. Em vez de hardcoded no Python, fica num arquivo legível e versionável.

### `configs/vlm_config.yaml` — backend custom

Três blocos:

- **`model:`** — qual encoder, qual LM, dimensões, hiperparâmetros LoRA.
  - `type: custom` é a chave que o `train.py` usa pra dispatch.
  - `vision_dim: 768` é a saída do CLIP ViT-B/32 (`pooler_output` shape `(B, 768)`).
  - `language_dim: 576` é o hidden size do SmolLM-135M (a Projection MLP precisa terminar nessa dimensão).
  - `lora_rank: 8`, `lora_alpha: 16` são os hiperparâmetros da decomposição low-rank.
- **`dataset:`** — `name` e `download` apontam pro dataset. `data/download.py` lê isso.
- **`training:`** — `epochs`, `batch_size`, `lr`, `warmup_steps`, `device`, `save_path`. CLI flags do `train.py` sobrescrevem isso.

### `configs/phi3_config.yaml` — backend phi3

Mesmos blocos, mas:
- `type: phi3` ativa o branch Phi-3 no factory de `train.py`.
- `load_in_4bit: true` ativa quantização NF4 via bitsandbytes.
- `batch_size: 2` é menor porque o modelo é 30× maior.
- Não tem `vision_encoder`/`projection_hidden_dim` — o Phi-3 já vem com vision embutido.

---

## `data/`

### `data/label_to_caption.py`

Converte arquivos YOLO `.txt` em frases.

**`_position(cx, cy) -> str`**
Recebe coordenadas normalizadas `[0, 1]` do centroide e retorna uma string posicional usando grid 3×3:
- `cx < 0.33` → `"left"`, `cx > 0.66` → `"right"`, senão `"center"`
- `cy < 0.33` → `"top"`, `cy > 0.66` → `"bottom"`, senão `"center"`
- Combina: `"top-left"`, `"bottom-right"`, etc. Quando uma coordenada cai no centro, simplifica: `(0.5, 0.1)` → `"top"` (não `"top-center"`).

**`yolo_labels_to_caption(label_path, class_names) -> str`**
Lê o arquivo `.txt` (uma linha por objeto: `class_id cx cy w h`), olha `class_names[class_id]` pra resolver o nome, e gera uma frase tipo `"A person at center. A car at bottom-left."`.

Edge cases tratados:
- Arquivo não existe → `"An image with no labeled objects."`
- Arquivo vazio → mesma string.
- Linha malformada (menos de 5 campos) → ignorada.
- Artigo indefinido escolhe `"An"` se a palavra começa com vogal, senão `"A"` (`An apple`, `A car`).

### `data/download.py`

Garante que o dataset existe localmente, baixando se necessário.

**`ensure_dataset(dataset_cfg: dict) -> Path`**
Recebe o bloco `dataset:` do YAML (com `name` e `download`). Fluxo:
1. Calcula `dest = ~/.cache/yolo-vlm/<name>/` e `yaml_path = dest/<name>.yaml`.
2. Se `yaml_path` já existe → retorna direto (idempotente).
3. Senão: baixa o `.zip` da URL com tqdm, extrai em `~/.cache/yolo-vlm/`, deleta o zip.
4. Gera o `<name>.yaml` apontando pra estrutura `images/`, `labels/`.

**`_write_coco8_yaml(root, yaml_path)`**
Escreve um YAML completo com os 80 nomes de classes do COCO. Hardcoded porque o zip do Ultralytics não vem com `.yaml`.

**`_write_yaml(name, root, yaml_path)`**
Dispatch: se for `coco8`, usa o template completo; senão escreve um YAML mínimo (`names: {}`) que o usuário precisa completar.

### `data/yolo_dataset.py`

Dataset PyTorch para o backend custom.

**`YOLOVLMDataset(yaml_path, split, processor, tokenizer, max_length=64)`**

No `__init__`:
1. Lê o YAML, resolve `dataset_root` (suporta path relativo).
2. Constrói lista de `(img_path, label_path)` varrendo `images/<split>/`.
3. Normaliza `class_names` pra dict `{int: str}` (aceita lista ou dict no YAML).

No `__getitem__(idx)`:
1. Abre a imagem com PIL, converte pra RGB.
2. Chama `yolo_labels_to_caption()` pra gerar a caption.
3. Pré-processa imagem com o `CLIPProcessor`: resize 224×224, normalização ImageNet → `pixel_values` shape `(3, 224, 224)`.
4. Tokeniza a caption com padding fixo até `max_length=64` → `input_ids` e `attention_mask` shape `(64,)`.
5. Retorna dict: `{pixel_values, input_ids, attention_mask}`.

O `DataLoader` empilha tudo em batch dim 0, virando shapes `(B, 3, 224, 224)`, `(B, 64)`, `(B, 64)`.

### `data/phi3_dataset.py`

Dataset PyTorch para o backend phi3.

A diferença chave: Phi-3 espera **chat template** com tokens especiais e **imagem inline** via `<|image_1|>`.

```python
_PROMPT = "<|user|>\n<|image_1|>\nDescribe the objects in this image.<|end|>\n<|assistant|>\n"
_SUFFIX = "<|end|>"
```

**`Phi3Dataset(yaml_path, split, processor, max_length=256)`**

No `__init__`: igual ao `YOLOVLMDataset` — varre imagens, monta lista, resolve nomes de classes.

No `__getitem__(idx)`:
1. Carrega imagem + gera caption (igual).
2. Monta `full_text = _PROMPT + caption + _SUFFIX`.
3. Chama o `processor` do Phi-3 com **imagem e texto juntos**. O processor faz três coisas:
   - Tokeniza o texto, **inserindo ~1000+ tokens de imagem** no lugar do `<|image_1|>`.
   - Pré-processa a imagem em patches.
   - Retorna `input_ids`, `attention_mask`, `pixel_values`, `image_sizes`.
4. Constrói `labels`:
   - Clone de `input_ids`.
   - Busca a posição do token `<|assistant|>` no `input_ids` real (que já tem os image tokens).
   - `response_start = position + 2` (pula o `<|assistant|>` e o `\n` seguinte).
   - `labels[:response_start] = -100` mascara prompt + image tokens.
   - `labels[labels == pad_token_id] = -100` mascara padding.

Resultado: o loss só é calculado nas posições da caption real. Tokens de imagem, prompt e padding são ignorados.

---

## `models/`

### `models/vision_encoder.py`

**`FrozenVisionEncoder(model_name)`**

Wrapper sobre `CLIPVisionModel` (HuggingFace). No `__init__`:
- Carrega o modelo pré-treinado.
- Marca todos os parâmetros com `requires_grad = False`.

No `forward(pixel_values)`:
- Decorado com `@torch.no_grad()` (não calcula gradientes, economiza memória).
- Retorna `outputs.pooler_output` shape `(B, 768)` — o embedding do token `[CLS]` final, que agrega informação da imagem inteira.

### `models/projection.py`

**`ProjectionMLP(vision_dim, hidden_dim, language_dim)`**

Sequential simples:
```
Linear(vision_dim → hidden_dim)
GELU()
Linear(hidden_dim → language_dim)
```

No `forward(x)`:
- `x` shape `(B, vision_dim)`.
- Passa pela MLP → `(B, language_dim)`.
- `unsqueeze(1)` → `(B, 1, language_dim)`. Esse `1` é a dimensão "sequence length" — estamos dizendo "este é UM token visual".

### `models/language_decoder.py`

**`LoRALanguageDecoder(model_name, lora_rank, lora_alpha, lora_dropout)`**

Wrapper que aplica LoRA num LM causal:
1. Carrega o modelo com `AutoModelForCausalLM.from_pretrained(model_name)`.
2. Define `LoraConfig` com `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (as 4 projeções de attention da arquitetura Llama, que é o que SmolLM usa).
3. `get_peft_model(base, lora_cfg)` envolve o modelo, congelando o original e adicionando os adaptadores treináveis.

**`get_input_embeddings()`** — expõe a embedding table interna do LM. Usada pelo `SmallVLM` pra converter `input_ids` em vetores que serão concatenados com o vision token.

**`forward(inputs_embeds, attention_mask, labels)`** — passa direto pro modelo PEFT. Note que **passamos `inputs_embeds`, não `input_ids`** — porque queremos prepender o vision token, que não vem da embedding table.

**`generate(inputs_embeds, attention_mask, max_new_tokens, **kwargs)`** — geração autoregressiva. `do_sample=False` força greedy decoding (sempre escolhe argmax).

### `models/small_vlm.py`

**`SmallVLM(cfg)`**

A pipeline custom completa. No `__init__`:
- Instancia `FrozenVisionEncoder`, `ProjectionMLP`, `LoRALanguageDecoder` a partir do `cfg["model"]`.
- Salva `max_new_tokens` pra geração.

**`_lm_dtype()`** — retorna o dtype dos parâmetros do decoder (tipicamente `bfloat16`). Usado pra cast.

**`_build_inputs_embeds(pixel_values, input_ids)`**
1. `vision_features = vision_encoder(pixel_values)` → `(B, 768)` em float32.
2. `vision_token = projection(vision_features).to(dtype)` → `(B, 1, 576)` em bfloat16.
3. `token_embeds = decoder.get_input_embeddings()(input_ids)` → `(B, T, 576)`.
4. `cat([vision_token, token_embeds], dim=1)` → `(B, 1+T, 576)`.

**`forward(pixel_values, input_ids, attention_mask)`**
1. Constrói `inputs_embeds` com o método acima → `(B, 1+T, 576)`.
2. **Mascara `labels` ANTES de extender o `attention_mask`** (shapes precisam bater pra boolean indexing):
   - `labels = input_ids.clone()` → `(B, T)`
   - `labels[attention_mask == 0] = -100` (padding ignorado no loss).
3. Estende `attention_mask` pra incluir o vision token: `(B, T)` → `(B, 1+T)` com `1` no início.
4. Estende `labels` com `-100` no início (posição do vision token, sem texto associado): `(B, T)` → `(B, 1+T)`.
5. `decoder(inputs_embeds, attention_mask, labels)` → retorna objeto com `.loss` (cross-entropy) e `.logits`.

**`generate(pixel_values, input_ids=None, attention_mask=None)`**
- Se `input_ids` for dado, gera continuando a partir dele (modo "completar texto").
- Se for `None`, gera só a partir do vision token (modo padrão de inferência: imagem → caption do zero).
- Constrói explicitamente um `attention_mask` de uns no caso vision-only, pra evitar warning do HuggingFace.

### `models/phi3_vlm.py`

**`Phi3VLM(cfg)`**

Wrapper sobre `microsoft/Phi-3-vision-128k-instruct`. No `__init__`:
1. Se `load_in_4bit: true`, monta `BitsAndBytesConfig`:
   - `load_in_4bit=True` ativa NF4.
   - `bnb_4bit_quant_type="nf4"` é o quant type específico.
   - `bnb_4bit_compute_dtype=bfloat16` — pesos são guardados em 4-bit, mas operações usam bfloat16.
   - `bnb_4bit_use_double_quant=True` quantiza também os parâmetros de escala (economiza mais bits).
2. Carrega o modelo com `trust_remote_code=True` (Phi-3 tem código custom no repo do HF), `torch_dtype=bfloat16`, e `_attn_implementation="eager"` (mais estável com 4-bit).
3. Aplica LoRA com `target_modules=["qkv_proj", "o_proj"]` — os módulos de attention do Phi-3 (note: `qkv_proj` é uma única matriz combinada em vez de 3 separadas).

**`forward(input_ids, attention_mask, pixel_values, image_sizes, labels)`** — passa direto pro modelo PEFT. O Phi-3 internamente lida com a fusão imagem+texto.

**`generate(input_ids, attention_mask, pixel_values, image_sizes)`** — geração com `do_sample=False`. `eos_token_id` explícito pro modelo saber quando parar.

---

## `train.py`

Loop de treino unificado que serve **os dois backends**.

**`build_components(cfg, data_yaml)`** — factory:
- Se `cfg["model"]["type"] == "phi3"`: instancia `Phi3VLM` + `Phi3Dataset`. Tokenizer é `processor.tokenizer`.
- Senão: instancia `SmallVLM` + `YOLOVLMDataset`. Tokenizer separado pra captions, processor separado pra imagens.

Retorna `(model, train_ds, val_ds, tokenizer)`.

**`parse_args()`** — argparse:
- `--data` opcional (path pra um YAML custom; default usa `dataset:` do config + auto-download).
- `--config` aponta pro YAML.
- `--epochs`, `--lr`, `--batch-size` sobrescrevem o YAML pra runs rápidas sem editar arquivo.

**`main()`**:
1. Carrega YAML.
2. Aplica overrides CLI.
3. Resolve dataset: `args.data or ensure_dataset(cfg["dataset"])`.
4. Resolve device: `cuda` se disponível, senão `cpu`.
5. Constrói modelo + datasets.
6. Move modelo pro device — **exceto** se `load_in_4bit=true`, porque bitsandbytes já posiciona durante o load.
7. Conta parâmetros treináveis (vai imprimir `~2M` em ambos os backends).
8. Cria DataLoaders (`num_workers=2`, shuffle no train).
9. AdamW só sobre os parâmetros com `requires_grad=True` (CLIP frozen é ignorado automaticamente).
10. `get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)` — total_steps é o produto de `len(train_loader) * epochs`, ou seja, **per-step**.
11. Loop de épocas:
    - Train: forward → loss → backward → clip_grad_norm → optimizer.step → scheduler.step → zero_grad. Prints a cada `log_every` steps.
    - Eval: `model.eval()` + `torch.no_grad()`, calcula val_loss.
    - Salva checkpoint: `{"epoch", "model_state": state_dict, "cfg": cfg}` em `vlm_epoch{N:02d}.pt`. O `cfg` salvo permite que `inference.py` reconstrua o modelo sem flags extras.

A passagem genérica `**batch_device` faz com que o **mesmo loop** funcione pros dois backends — cada `Dataset` retorna um dict com as keys que o respectivo `forward` espera.

---

## `inference.py`

Geração de caption pra uma imagem isolada.

**`_latest_checkpoint(ckpt_dir)`** — `sorted(glob("*.pt"))[-1]`. Funciona porque os nomes têm zero-padding (`vlm_epoch01.pt` vs `vlm_epoch10.pt`).

**`parse_args()`** — image (posicional), `--checkpoint` (opcional, default = mais recente).

**`main()`**:
1. Resolve checkpoint path.
2. `torch.load(ckpt_path, weights_only=False)` — `False` é necessário pra restaurar o `cfg` (dict Python).
3. `cfg = ckpt["cfg"]`, `model_type = cfg["model"]["type"]`.
4. Abre imagem com PIL.
5. **Dispatch por `model_type`**:
   - `"phi3"`: carrega `AutoProcessor` do Phi-3, instancia `Phi3VLM`, carrega state_dict, processa `_PROMPT` + imagem juntos, gera, decodifica **só os tokens novos** (`output_ids[0][input_ids.shape[1]:]`).
   - `"custom"`: carrega `CLIPProcessor` + `AutoTokenizer` do SmolLM, instancia `SmallVLM`, processa só a imagem (sem texto inicial), gera, decodifica tudo.
6. Print da caption.

A grande sacada é que **o usuário não precisa saber qual backend foi treinado** — o checkpoint salva o `cfg` e o `inference.py` se adapta sozinho.

---

## Fluxo End-to-End — Custom Backend

Vamos rastrear uma sample do disco até o gradiente de uma única atualização de pesos.

```
disco
  ↓
[YOLOVLMDataset.__getitem__]
  - lê imagem 640×480 PNG
  - lê labels: "0 0.5 0.4 0.2 0.6"
  ↓
[label_to_caption]
  caption = "A person at center."
  ↓
[CLIPProcessor]
  pixel_values: (3, 224, 224) float32
[SmolLM tokenizer, padding=64]
  input_ids: (64,)         int64
  attention_mask: (64,)    int64
  ↓
[DataLoader collate, batch_size=8]
  pixel_values: (8, 3, 224, 224)
  input_ids: (8, 64)
  attention_mask: (8, 64)
  ↓
[SmallVLM.forward]
  ├── FrozenVisionEncoder(pixel_values) → (8, 768) float32
  ├── ProjectionMLP(vf) → (8, 1, 576) → cast bf16
  ├── decoder.embed(input_ids) → (8, 64, 576) bf16
  ├── cat → inputs_embeds: (8, 65, 576)
  ├── attention_mask: (8, 64) → cat ones → (8, 65)
  ├── labels: (8, 64)
  │   - mask onde attention_mask==0 (padding) → -100
  │   - prepend -100 → (8, 65)
  └── LoRALanguageDecoder(inputs_embeds, attention_mask, labels)
      → logits: (8, 65, vocab_size)
      → loss: scalar (cross-entropy só nas posições != -100)
  ↓
loss.backward()
  - autograd preenche .grad em cada param treinável
  - LoRA matrices A, B (~1M params)
  - Projection MLP weights (~1M params)
  - CLIP, SmolLM base: requires_grad=False, ignorados
  ↓
clip_grad_norm_(trainable, 1.0)
optimizer.step()    # AdamW atualiza apenas trainable
scheduler.step()    # cosine schedule avança 1 step
optimizer.zero_grad()
```

---

## Fluxo End-to-End — Phi-3 Backend

```
disco
  ↓
[Phi3Dataset.__getitem__]
  caption = "A person at center."
  full_text = "<|user|>\n<|image_1|>\nDescribe...<|end|>\n<|assistant|>\nA person at center.<|end|>"
  ↓
[Phi3 processor]
  - tokeniza texto, expande <|image_1|> em ~1000 image tokens
  - processa imagem em patches
  input_ids: (256,)  ← ~22 prompt + ~1000 image + caption + padding
  attention_mask: (256,)
  pixel_values: tensor de patches
  image_sizes: (2,)
  ↓
[label masking]
  - busca <|assistant|> em input_ids → posição N
  - labels[:N+2] = -100   # mascara prompt + imagem + <|assistant|> + \n
  - labels[labels == pad_token_id] = -100
  ↓
[DataLoader collate, batch_size=2]
  ↓
[Phi3VLM.forward]
  - delegado direto pro Phi-3 base com LoRA
  - vision encoding interno → fusão com texto → causal LM
  loss: scalar (cross-entropy só nas posições da caption real)
  ↓
backward + AdamW + scheduler  (idêntico ao custom)
```

---

## Decisões de Design Importantes

### Por que `cfg` salvo no checkpoint?

`torch.save({"model_state", "cfg"})` em vez de só `state_dict`. O `cfg` permite reconstruir o modelo com a arquitetura exata sem o usuário precisar saber dos hiperparâmetros. Custo: alguns KBs no `.pt`. Benefício: reprodutibilidade total.

### Por que `**batch_device`?

`outputs = model(**batch_device)` no loop de treino faz com que o mesmo código funcione pros dois backends. Cada `Dataset` retorna um dict com as keys que o respectivo `forward` espera (`{pixel_values, input_ids, attention_mask}` pro custom; `{input_ids, attention_mask, pixel_values, image_sizes, labels}` pro phi3). Zero `if model_type == ...` no loop.

### Por que `pad_token = eos_token` no SmolLM?

SmolLM não vem com pad token definido. Sem isso, o tokenizer crasha quando `padding="max_length"`. Reusar o `eos_token` é prática padrão no HuggingFace — só importa que o `attention_mask` esteja correto (e está, porque o tokenizer marca padding como 0 mesmo que o token seja eos_id).

### Por que `weights_only=False` no `torch.load`?

Por padrão, novas versões do PyTorch usam `weights_only=True` (carrega só tensores, mais seguro). Como nosso checkpoint contém um dict `cfg` Python, precisamos do modo "unsafe". O risco existe só se você carregar checkpoints de fontes não confiáveis.

### Por que `_attn_implementation="eager"` no Phi-3?

A implementação SDPA/Flash attention pode ter inconsistências com 4-bit quantization. "eager" usa attention manual em PyTorch puro — mais lento, mas estável.

---

## Resumo: Quem Treina o Quê?

| Componente | Custom | Phi-3 |
|---|---|---|
| Vision encoder | CLIP frozen (87M, 0 trainable) | Embutido no Phi-3, frozen via base 4-bit |
| Projection | MLP treinada (~1M trainable) | N/A — Phi-3 já tem fusão interna |
| Language model | SmolLM base frozen + LoRA (~1M trainable) | Phi-3 base 4-bit frozen + LoRA (~2M trainable) |
| Total trainable | ~2M | ~2M |
| VRAM em treino | ~2 GB | ~3 GB (4-bit) / ~8 GB (bf16) |

Em ambos os casos, a **lógica de treino** é a mesma — backprop através do que tem `requires_grad=True`, mas a **maior parte do modelo fica congelada**. Isso é o que torna fine-tuning de VLMs viável em GPUs de consumidor.

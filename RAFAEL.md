# RAFAEL — A Lógica Científica por Trás do Small VLM

Este documento explica **o que está acontecendo** quando o modelo treina e inferencia. Não fala de código — fala de princípios: representações vetoriais, espaços de embedding, otimização, cross-entropy, fine-tuning eficiente. Se você entender isso, entende qualquer VLM.

---

## 1. O Problema Fundamental

Um Vision-Language Model (VLM) precisa **traduzir pixels em palavras**. Mas pixels e palavras vivem em universos representacionais diferentes:

- Pixels são tensores `(H × W × 3)` de intensidades RGB. Não têm semântica intrínseca — `[127, 200, 50]` não significa "verde claro" pro modelo, significa só três números.
- Palavras vivem como **token IDs discretos** que indexam uma embedding table. O token "person" vira um vetor `(d,)` aprendido durante o pré-treino do LM.

A grande pergunta: **como fazer um decoder de linguagem (que só sabe trabalhar com sequências de embeddings de tokens) "ler" uma imagem?**

A resposta — e o coração de todo VLM — é **alinhar os dois espaços**: representar a imagem como um vetor que vive *dentro* do mesmo espaço de embedding que o LM, como se fosse "mais um token".

---

## 2. Visão: Como CLIP Transforma uma Imagem em um Vetor Semântico

CLIP (Contrastive Language-Image Pre-training, OpenAI 2021) é o vision encoder usado no backend custom. Ele foi pré-treinado em **400 milhões de pares imagem-texto** com o seguinte objetivo:

> Dado um batch de N imagens e N legendas, faça com que cada imagem fique mais próxima da sua legenda correta do que de qualquer outra das N-1 legendas.

Matematicamente, isso é **InfoNCE loss**:

```
L = -log(exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))
```

onde `sim` é cosine similarity e `τ` é uma temperatura. O efeito: imagens com semântica parecida acabam em regiões próximas do espaço, e suas legendas também — mas mais importante, **a região da imagem fica alinhada com a região do texto correspondente**.

### Por que CLIP é especial

O encoder de visão do CLIP não foi treinado em ImageNet pra classificar 1000 categorias fixas. Foi treinado pra **representar conceitos abertos**. O vetor de saída de uma imagem de cachorro fica perto do vetor de saída do texto "a photo of a dog" — e isso vale pra **qualquer conceito**, não só os 1000 da ImageNet.

Quando passamos uma imagem pelo CLIP ViT-B/32:
1. A imagem é dividida em patches 32×32 (~49 patches numa imagem 224×224).
2. Cada patch vira um token via projeção linear.
3. Um token especial `[CLS]` é prependado.
4. Self-attention é aplicada por 12 camadas — cada token "olha" pros outros e refina sua representação.
5. O `[CLS]` final agrega informação global da imagem em um vetor `(768,)`.

Esse vetor de 768 dimensões é uma **assinatura semântica densa** da imagem. Cachorros caem perto de cachorros, carros perto de carros, e ambos longe de cenas urbanas.

### Por que congelamos o CLIP

CLIP já foi treinado em 400M de pares — o conhecimento visual dele é vasto. Treinar com COCO8 (apenas 8 imagens!) destruiria essa representação por **catastrophic forgetting**: gradientes pequenos amplificados ao longo de milhões de parâmetros causariam drift sem ganho. **Não treinamos vision quando temos um encoder pré-treinado bom** — só usamos.

---

## 3. Linguagem: Como SmolLM Gera Texto

SmolLM-135M é um **decoder-only transformer causal** (mesma família do GPT). "Causal" significa que cada token só pode atender aos tokens anteriores — não vê o futuro. Isso é essencial pra geração autoregressiva: o modelo prediz o próximo token, alimenta de volta, prediz o próximo, e assim por diante.

### O objetivo do treino: maximum likelihood

Dado uma sequência `(x_1, x_2, ..., x_T)`, o modelo aprende a maximizar:

```
P(x_1, x_2, ..., x_T) = Π P(x_t | x_1, ..., x_{t-1})
```

A loss é **cross-entropy** entre o token previsto e o token real:

```
L = -Σ log P(x_t | x_<t)
```

Em treino, isso é feito com **teacher forcing**: a entrada é a sequência real shiftada, o target é a sequência real, e o loss é calculado em todas as posições simultaneamente (paralelizável). Em inferência, o modelo gera token por token, alimentando a saída de volta.

### Por que decoder-only ao invés de encoder-decoder?

Encoder-decoder (T5, BART) tem dois passes: codificar entrada, decodificar saída. Decoder-only é mais simples — tudo é uma sequência só. Para VLMs, a vantagem é que basta **prepender** o vetor de visão à sequência de tokens, e o decoder trata tudo uniformemente.

---

## 4. A Ponte: Projection MLP

CLIP produz um vetor de 768 dimensões. SmolLM espera tokens de 576 dimensões. Os dois espaços **não são compatíveis** — não estão alinhados, não vivem na mesma "geometria".

A **Projection MLP** é uma rede neural pequena (2 camadas, ~1M parâmetros) que aprende a função:

```
f: R^768 → R^576
```

mapeando vetores do espaço CLIP pro espaço de embedding do SmolLM. Essa MLP é **a única parte da pipeline visual que treina** no backend custom. Ela aprende, por exemplo, que o conceito visual "person" do CLIP deve ser mapeado pra uma região do espaço SmolLM próxima do embedding do token "person".

### Por que uma MLP simples basta?

Há um resultado teórico: redes neurais com pelo menos uma camada hidden e ativação não-linear são **aproximadores universais** (Hornik 1991). Não precisamos de nada complicado pra alinhar dois espaços densos — uma `Linear → GELU → Linear` resolve.

### Por que GELU e não ReLU?

GELU (Gaussian Error Linear Unit) é a ativação padrão em transformers modernos. É como ReLU, mas suave — multiplica `x` por `Φ(x)` (CDF da normal). Empiricamente treina melhor em modelos baseados em attention.

---

## 5. A Concatenação Mágica: O Vision Token

Aqui acontece a fusão. Após projeção, temos:
- `vision_token`: shape `(1, 576)` — um vetor que vive no espaço SmolLM
- `text_tokens`: shape `(T, 576)` — embeddings da caption tokenizada

Concatenamos: `(1+T, 576)`. Pra SmolLM, é uma sequência de `1+T` tokens — o primeiro é "estranho" (não veio da embedding table), mas é exatamente do tipo certo pra ser processado.

O attention do SmolLM agora pode fazer algo poderoso: **atender ao vision token enquanto gera cada palavra da caption**. Quando ele prevê "person", a query do token "person" pode atender ao vision token e dizer: "este vetor visual contém uma pessoa, então 'person' é uma boa predição".

### Label masking: o truque do `-100`

PyTorch's CrossEntropyLoss ignora posições com label `-100`. Usamos isso pra **dizer ao modelo onde NÃO calcular loss**:

- Posição 0 (vision token): `-100`. Não tem texto associado, não predizemos nada aqui.
- Posições de padding: `-100`. Padding é token "fake" pra alinhar batch sizes — não queremos otimizar pra prever padding.
- Posições reais da caption: token IDs verdadeiros.

Sem isso, o modelo aprenderia a prever padding, o que é inútil e contamina a loss.

---

## 6. LoRA — A Mágica do Fine-Tuning Eficiente

SmolLM-135M tem 135 milhões de parâmetros. Treinar todos eles em COCO8 (8 imagens) seria absurdo — overfit catastrófico. A solução é **LoRA** (Low-Rank Adaptation, Hu et al. 2021).

### A intuição

A maioria dos pesos de um LM pré-treinado já é boa — ele já sabe gerar inglês fluente. O que precisamos é uma **pequena correção** pra que ele aprenda a usar o vision token. Em vez de ajustar a matriz de pesos `W ∈ R^(d×d)` inteira, decompomos a correção em:

```
W' = W + ΔW = W + B·A    onde A ∈ R^(r×d), B ∈ R^(d×r), r << d
```

Com `r=8` e `d=576`, em vez de treinar `576×576 = 332k` parâmetros por matriz, treinamos `2×576×8 = 9.2k` — **36× menos**. E como aplicamos LoRA só nas camadas de attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), o total trainable é ~1M parâmetros pra um modelo de 135M.

### Por que LoRA funciona

Pesquisa empírica mostra que o **delta** entre o modelo pré-treinado e o fine-tuned tem **rank baixo intrínseco** — ou seja, a correção necessária é "pequena" em alguma base. LoRA explora isso explicitamente: força a correção a ser baixa-rank `r`, o que regulariza o modelo (impede overfitting) e reduz drasticamente compute/memória.

### Os hiperparâmetros: rank e alpha

- `lora_rank=8`: dimensão da decomposição. Maior = mais capacidade, mais risco de overfit.
- `lora_alpha=16`: escala da contribuição. O update efetivo é `(alpha/rank)·B·A = 2·B·A`. Manter `alpha = 2·rank` é convenção comum.

---

## 7. YOLO Bounding Boxes → Linguagem Natural

O dataset YOLO dá:
```
0 0.50 0.40 0.20 0.58
```
que significa: classe 0 (person), centroide em (0.50, 0.40), largura 0.20, altura 0.58 — tudo normalizado [0,1].

O modelo de linguagem **não entende coordenadas**. Tem que virar texto. A solução é uma **discretização espacial**:

- Dividimos a imagem em grid 3×3.
- Cada centroide cai em uma das 9 células: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right.

`(0.50, 0.40)` → coluna "center", linha "center" → "center".
`(0.10, 0.80)` → coluna "left", linha "bottom" → "bottom-left".

E geramos uma frase: `"A person at center. A car at bottom-left."`

### Por que isso é uma supervisão razoável

Não estamos ensinando o modelo a localizar precisamente — só a **descrever**. Posições aproximadas em palavras são informação genuína: dizer "a person at center" é mais informativo que só "a person". O modelo aprende a correlacionar regiões da imagem (via attention sobre o vision token) com linguagem posicional.

---

## 8. Os Dois Backends: Filosofias Diferentes

### Custom: Composição Modular

```
[Imagem] → CLIP → MLP → [Vision Token] → SmolLM → [Texto]
```

Três módulos separados, costurados manualmente. Vantagens:
- **Transparente**: você controla cada camada.
- **Leve**: ~2GB VRAM, ~2M parâmetros treináveis.
- **Pedagógico**: você vê como cada parte funciona.

Desvantagens:
- **Vision e linguagem foram pré-treinados em domínios diferentes.** A MLP de projeção tem que fazer todo o trabalho pesado de alinhamento.
- **Apenas um vision token.** Toda informação visual é comprimida em 576 dimensões — perdemos detalhes.

### Phi-3-vision: VLM End-to-End

```
[Imagem + Texto] → Phi-3 → [Texto]
```

Phi-3-vision foi **co-treinado em multimodal**: a Microsoft pré-treinou o modelo todo em pares imagem-texto, então o vision encoder, projection e language model foram otimizados juntos. Vantagens:
- **Alinhamento nativo** — não há "costura externa".
- **Múltiplos tokens visuais** (~1000) preservam detalhes.
- **Qualidade dramaticamente melhor** em descrição, OCR, raciocínio visual.

Desvantagens:
- **4.2B parâmetros** — precisa quantização pra rodar em GPUs comuns.
- **Caixa preta**: você não escolhe o vision encoder, a projeção, nada.

### A escolha: aprender vs usar

O backend custom é pedagógico — você entende cada peça. O Phi-3 é prático — você produz resultado SOTA com poucas linhas. Os dois são valiosos por motivos diferentes.

---

## 9. Quantização 4-bit (NF4)

Phi-3 4.2B params em float32 ocupa `4.2B × 4 bytes = 16.8 GB`. Em bfloat16, `8.4 GB` — ainda demais pra muitas GPUs. **NF4** (NormalFloat 4-bit, Dettmers et al. 2023) reduz pra ~3 GB.

### Como funciona

Em vez de 32 bits por peso (4 bilhões de valores possíveis), usamos 4 bits (16 valores). Mas não usamos uniformemente — usamos uma **escala não-linear baseada na distribuição normal**: pesos de redes neurais seguem aproximadamente N(0, σ²), então é mais informativo ter resolução fina perto de zero (onde a maioria está) e grossa nas caudas.

### Compute em bfloat16, storage em 4-bit

Quando o forward acontece, os pesos 4-bit são **descomprimidos on-the-fly** pra bfloat16, multiplicados, e descartados. Compute é em bfloat16 (preciso), storage é em 4-bit (compacto). É como streaming de áudio comprimido — você ouve em qualidade, mas armazena pequeno.

### Double quantization

Os parâmetros de escala (que controlam a descompressão) também são quantizados. Reduz mais alguns bits por peso. NF4 + double quant + bfloat16 compute = 4.2B params em 3 GB.

### Por que LoRA é essencial com 4-bit

Pesos quantizados não são treináveis — descomprimir, treinar e re-comprimir destruiria a qualidade. LoRA contorna isso: o modelo base fica congelado em 4-bit, e os adaptadores LoRA (em bfloat16, full precision) é o que treinamos. Os pesos 4-bit dão capacidade representacional, os adaptadores LoRA dão adaptação.

---

## 10. O Otimizador: AdamW + Cosine Schedule + Warmup

Treinamento neural é **descida de gradiente em loss landscape de altíssima dimensão**. A escolha de otimizador e schedule de learning rate é crítica.

### AdamW

Adam mantém duas estatísticas por parâmetro:
- **Momento de primeira ordem (média móvel do gradiente)**: dá direção persistente.
- **Momento de segunda ordem (média móvel do gradiente ao quadrado)**: dá magnitude adaptativa por parâmetro — gradientes pequenos recebem step grande, gradientes grandes recebem step pequeno.

AdamW é Adam com **weight decay desacoplado**: penaliza norma dos pesos sem misturar com o estado do otimizador. Mais correto matematicamente que Adam original com L2 reg.

### Cosine Schedule with Warmup

Treinamento começa com **warmup linear**: a learning rate sobe de 0 até `lr_max` ao longo dos primeiros N steps. Por quê? Porque os momentos do Adam ainda são imprecisos no início — ataques agressivos com lr alta destabilizam.

Depois do warmup, a lr **decai como cossseno** de `lr_max` até 0:

```
lr(t) = lr_max · 0.5 · (1 + cos(π · t / T))
```

A intuição: **early training precisa explorar** (lr alta encontra regiões interessantes do landscape), **late training precisa convergir** (lr baixa refina o mínimo). Cosine é uma curva suave que naturalmente passa de uma fase pra outra.

### Gradient Clipping

`clip_grad_norm_(params, 1.0)` limita a norma do gradiente a 1.0 antes do step. Por quê? Em transformers, ocasionalmente um batch ruim produz gradientes enormes que explodem os pesos (NaN training). Clipping é uma rede de segurança.

---

## 11. O Que Acontece em Uma Iteração de Treino

1. **Forward**: imagem + caption entram. Cross-entropy loss é calculada nas posições não-mascaradas.
2. **Backward**: PyTorch autograd calcula `∂L/∂θ` pra cada parâmetro treinável (LoRA + Projection).
3. **Clip**: norma do gradiente é cortada em 1.0.
4. **Step**: AdamW atualiza os parâmetros: `θ ← θ - lr · adam_update(grad)`.
5. **Schedule**: cosine schedule avança um step (lr decai um pouquinho).
6. **Zero grad**: gradientes acumulados são zerados pro próximo batch.

Repetimos isso `epochs × batches_per_epoch` vezes. Cada epoch passa por todo o dataset uma vez.

---

## 12. Inferência: Geração Autoregressiva

No inference time, não temos caption — temos só a imagem. O fluxo:

1. Imagem → CLIP → MLP → vision_token `(1, 576)`.
2. Decoder recebe **só** o vision_token como `inputs_embeds`. Sem texto inicial.
3. Decoder prediz a distribuição do próximo token: `P(x_1 | vision)`.
4. Escolhemos o token mais provável (greedy, `do_sample=False`).
5. Convertemos esse token em embedding, concatenamos: `[vision, x_1]`.
6. Predizemos `P(x_2 | vision, x_1)`. Repetimos até gerar `<eos>` ou atingir `max_new_tokens`.

### Por que greedy decoding aqui

`do_sample=False` significa: sempre escolha o argmax. Determinístico, reproduzível, suficiente pra captions curtas factuais. Sampling (top-k, top-p, temperature) seria útil pra criatividade, mas em descrição de objetos não queremos variação — queremos a descrição mais provável.

---

## 13. Por Que Tudo Isso Faz Sentido

Junte as peças:

- **CLIP** te dá um espaço onde imagens e textos são vizinhos quando significam coisas parecidas.
- **MLP de projeção** alinha esse espaço com o espaço de tokens do LM.
- **LM com LoRA** aprende a usar esse vetor projetado como contexto pra gerar texto.
- **Cross-entropy + teacher forcing** ensina o modelo a maximizar a probabilidade da caption certa dado a imagem.
- **AdamW + cosine schedule** otimiza isso em uma loss landscape complexa de forma estável.
- **YOLO → captions** transforma supervisão de bounding box em supervisão de linguagem natural.

O resultado é um modelo que, dado pixels, **gera texto** descrevendo o conteúdo. Não é mágica — é alinhamento de espaços vetoriais, otimização gradiente, e uma boa dose de pre-training prévio.

Phi-3 faz a mesma coisa, mas com tudo pré-alinhado de fábrica e em escala 30× maior. A ideia é a mesma. A diferença é só de magnitude.

---

## 14. Limitações Conscientes

- **8 imagens (COCO8) é absurdamente pouco**. O modelo vai memorizar, não generalizar. COCO8 é pra teste de pipeline, não pra ter qualidade real. Pra ter um modelo útil, precisaria de COCO128k+ ou Visual Genome.
- **Captions são templates rígidos** ("A {class} at {position}."). O modelo aprende esse padrão, não linguagem livre.
- **Um único vision token (custom)** é um gargalo de informação. VLMs reais (LLaVA, Phi-3) usam dezenas/centenas.
- **Greedy decoding** pode ficar repetitivo. Beam search ou nucleus sampling melhoraria fluência.

São limitações do escopo do projeto, não bugs. O propósito é **didático**: entender a anatomia de um VLM end-to-end, não competir com GPT-4V.

---

## Referências

### Vision-Language e Contrastive Learning

- **CLIP** — Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. arXiv:[2103.00020](https://arxiv.org/abs/2103.00020)
- **InfoNCE** — van den Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:[1807.03748](https://arxiv.org/abs/1807.03748)
- **ViT (Vision Transformer)** — Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv:[2010.11929](https://arxiv.org/abs/2010.11929)
- **Phi-3-vision** — Abdin et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:[2404.14219](https://arxiv.org/abs/2404.14219)
- **LLaVA** (referência conceitual de VLMs com vision token) — Liu et al. (2023). *Visual Instruction Tuning*. arXiv:[2304.08485](https://arxiv.org/abs/2304.08485)

### Language Modeling

- **Transformer** — Vaswani et al. (2017). *Attention Is All You Need*. arXiv:[1706.03762](https://arxiv.org/abs/1706.03762)
- **GPT-2 / decoder-only causal LM** — Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*. [OpenAI tech report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Llama (arquitetura usada por SmolLM)** — Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:[2302.13971](https://arxiv.org/abs/2302.13971)
- **SmolLM** — HuggingFace blog (2024). *SmolLM - blazingly fast and remarkably powerful*. https://huggingface.co/blog/smollm

### Fine-Tuning Eficiente

- **LoRA** — Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:[2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA / NF4 quantization** — Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:[2305.14314](https://arxiv.org/abs/2305.14314)
- **bitsandbytes (8-bit base)** — Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. arXiv:[2208.07339](https://arxiv.org/abs/2208.07339)

### Otimização

- **Adam** — Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization*. arXiv:[1412.6980](https://arxiv.org/abs/1412.6980)
- **AdamW** — Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization*. arXiv:[1711.05101](https://arxiv.org/abs/1711.05101)
- **Cosine annealing / SGDR** — Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:[1608.03983](https://arxiv.org/abs/1608.03983)
- **Linear warmup** — Goyal et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:[1706.02677](https://arxiv.org/abs/1706.02677)

### Ativações e Fundamentos

- **GELU** — Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:[1606.08415](https://arxiv.org/abs/1606.08415)
- **Universal Approximation Theorem** — Hornik (1991). *Approximation Capabilities of Multilayer Feedforward Networks*. *Neural Networks*, 4(2):251–257.
- **Cross-entropy & Maximum Likelihood** — base estatística clássica; tratamento moderno em Goodfellow et al. (2016). *Deep Learning*. MIT Press, [www.deeplearningbook.org](https://www.deeplearningbook.org)

### Datasets

- **COCO** — Lin et al. (2014). *Microsoft COCO: Common Objects in Context*. arXiv:[1405.0312](https://arxiv.org/abs/1405.0312)
- **YOLO format** — Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. arXiv:[1506.02640](https://arxiv.org/abs/1506.02640). Formato canônico mantido em [Ultralytics docs](https://docs.ultralytics.com/datasets/detect/).

### Ferramentas

- **PEFT (HuggingFace)** — Mangrulkar et al. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods*. https://github.com/huggingface/peft
- **Transformers (HuggingFace)** — Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. https://github.com/huggingface/transformers

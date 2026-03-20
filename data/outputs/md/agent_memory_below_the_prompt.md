# 1. Introduction

---

### 1. 배경 및 문제 제기: Edge 디바이스에서 멀티 에이전트 LLM의 메모리 병목

- **멀티 에이전트의 메모리 문제:** Apple M4 Pro(24GB RAM) 기준, 10.2GB의 KV cache 예산에서 FP16 포맷으로는 **4K context에 에이전트 6개, 8K context에는 3개**만 동시 유지할 수 있습니다. 10개의 에이전트를 운용하려면 끊임없이 eviction과 reload를 반복해야 합니다.
- **Prefill 비용이 치명적:** 에이전트가 evict된 후 다시 활성화될 때마다 전체 context를 처음부터 다시 계산하는 **cold prefill**이 필요합니다. Gemma 3 12B 기준 4K context의 cold prefill은 **15.7초**가 걸립니다. 에이전트 5개가 서버 재시작 후 모두 cold-start를 해야 한다면 5 × 15.7 = **78.5초의 데드타임**이 발생합니다.
- **기존 접근법의 한계:**
  - **RAG:** 매 요청마다 텍스트 청크를 re-retrieve하고 prefill을 다시 수행하므로 $O(n)$ 연산 비용이 그대로 남습니다. RAG 추론 시간의 95.5%가 prefill입니다.
  - **vLLM/SGLang의 prefix caching:** FP16 KV cache를 메모리에 보관하므로, 서버 재시작 시 모든 캐시가 유실됩니다. 또한 에이전트 수가 늘어나면 메모리 pressure로 인해 캐시가 evict됩니다.
  - **프롬프트 결합(Concatenation):** 여러 에이전트의 대화를 하나의 긴 프롬프트로 합치면 **position bias** 문제가 발생하여 중간 부분의 정보가 무시됩니다.

### 2. 제안: Persistent Q4 KV Cache

핵심 아이디어는 간단합니다: **각 에이전트의 KV cache를 4-bit 양자화하여 디스크에 영속적으로 저장하고, 필요할 때 attention layer에 직접 로딩**합니다.

→ **$O(n)$ compute-bound prefill을 $O(n)$ I/O-bound cache reload로 대체하여, 15.7초의 재계산을 577ms의 디스크 읽기로 줄인다.**

- **Q4 양자화:** FP16 대비 **72% 메모리 절감** → 동일 예산에 4배 더 많은 에이전트 수용
- **디스크 영속성:** safetensors 포맷으로 SSD에 저장 → 서버 재시작, 디바이스 슬립/웨이크 후에도 캐시 유지
- **에이전트별 격리:** Block Pool 구조로 각 에이전트의 캐시를 독립 관리 → 보안(PROMPTPEEK 공격 방지) + GDPR/HIPAA 컴플라이언스 대응

### 3. 시스템 구성요소 및 핵심 결과

시스템은 세 가지 핵심 컴포넌트로 구성됩니다:

1. **Block Pool:** 에이전트별 격리된 Q4 KV cache를 safetensors 파일로 관리
2. **BatchQuantizedKVCache:** 여러 에이전트의 양자화된 캐시 위에서 동시 추론 지원
3. **Cross-Phase Context Injection:** 대화 페이즈를 넘나들며 attention state를 재계산 없이 누적

**실험 결과 (3가지 아키텍처에서 검증):**

| 모델 | TTFT 감소 (32K context) | Perplexity 변화 |
|---|---|---|
| Gemma 3 12B (GQA, 48 layers) | 최대 **136×** (172초 → 1.3초) | −0.7% (측정 노이즈 범위) |
| DeepSeek-Coder-V2-Lite 16B (MoE, MLA) | 최대 **76×** (47.3초 → 624ms) | +3.0% |
| Llama 3.1 8B (GQA, 32 layers) | 최대 **111×** (47.6초 → 526ms, 16K) | +2.8% |

# 2. System Design

---

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_1.png)

위 다이어그램은 시스템의 전체 아키텍처를 보여줍니다. 왼쪽의 여러 에이전트(Code Assistant 4K, Reviewer 2K, Planner 3K)가 각각 독립적인 context 길이를 가지고 **Block Pool**에 접근합니다. Block Pool은 에이전트별 격리를 보장하며, **Q4 Pipeline**을 통해 양자화 및 attention 연산이 수행됩니다. 오른쪽의 디스크(safetensors)에 캐시를 영속적으로 저장하고, 50ms 수준으로 reload합니다.

→ **핵심 성과: 30–130× TTFT 감소, 72% 메모리 절감, 1초 미만의 캐시 reload**

## 2.1 Block Pool: 에이전트별 격리

Block Pool은 KV cache를 **256 토큰 단위의 고정 크기 블록**으로 분할하여 에이전트 ID별로 관리합니다.

- 각 에이전트의 캐시는 **AgentBlocks** 자료구조로 구성: 레이어 인덱스 → KVBlock 리스트 매핑
- 각 KVBlock은 Q4 포맷(uint32 packed data + bfloat16 scales/biases)으로 Key/Value 텐서를 레이어별 저장
- **ModelCacheSpec** 추상화가 모델별 차이(레이어 수, KV head 수, head dimension, 양자화 설정)를 캡슐화

> 서버 재시작, 모델 교체, 동시 추론 등 어떤 상황에서도 에이전트 간 캐시가 오염되지 않습니다. PROMPTPEEK 공격(공유 KV cache에서 다른 사용자 프롬프트를 99% 복원)을 구조적으로 차단합니다.

## 2.2 Q4 Quantization Pipeline

KV cache는 시스템 전체에서 **일관되게 4-bit 양자화 포맷**으로 유지됩니다:

1. **Disk:** uint32 packed data + bfloat16 scales/biases → safetensors 포맷
2. **Memory:** 동일 포맷, `mx.load()`로 로딩
3. **Attention:** mlx-lm의 `quantized_scaled_dot_product_attention()`이 **Q4 텐서에 직접 연산**

**FP16 대비 메모리 비율:** group size $g=64$ 기준, $\text{Q4/FP16} = (1 + 8/g)/4 = 0.281$ → **72% 메모리 절감**

| Context | FP16/agent | Q4/agent | FP16 수용 | Q4 수용 |
|---|---|---|---|---|
| 4K | 1.5 GB | 0.42 GB | 6 | 24 |
| 8K | 3.0 GB | 0.84 GB | 3 | 12 |
| 16K | 6.0 GB | 1.7 GB | 1 | 6 |
| 32K | 12.0 GB | 3.4 GB | 0 | 3 |

→ **8K context에서 FP16은 3개, Q4는 12개 에이전트를 수용합니다. 4배 차이입니다.**

## 2.3 Prefix Matching: 문자 단위 비교

기존 prefix caching 시스템(vLLM, SGLang)은 **토큰 ID를 비교**합니다. 하지만 BPE 토크나이저는 context-dependent하여 동일한 텍스트가 주변 토큰에 따라 다른 토큰 시퀀스를 생성할 수 있습니다.

이 시스템은 **원본 텍스트를 문자(character) 단위로 비교**하여 세 가지 결과를 반환합니다:
- **EXACT:** 완전 일치 → 캐시 전체 재사용
- **EXTEND:** 새 프롬프트가 캐시된 텍스트로 시작 → 캐시 재사용 + 새 부분만 prefill
- **DIVERGE:** 공통 prefix 부족 → 캐시 재사용 불가

> 멀티 페이즈 에이전트 워크플로우에서 프롬프트는 대부분 **단조 증가(monotonically growing)**하므로, 실질적으로 EXTEND 매칭이 주로 발생합니다.

## 2.4 Batched Quantized Inference

MLX 업스트림 라이브러리(mlx-lm v0.30)는 양자화된 KV cache에 대한 batched inference를 지원하지 않습니다. 이 시스템은 **BatchQuantizedKVCache**를 구현하여 세 가지 연산을 제공합니다:

- **merge:** 에이전트별 캐시를 left-pad하여 batch 차원으로 스택
- **update_and_fetch:** 통합 batch에서 attention 계산 및 새 KV 쌍 업데이트
- **extract:** batch 결과를 다시 에이전트별 캐시로 분리

**ConcurrentScheduler**가 prefill 청크(기본 512 토큰)와 decode 스텝을 인터리빙하여, Orca의 iteration-level 스케줄링을 edge 디바이스의 양자화 캐시에 적용합니다.

> **Concurrency 모델:** MLX는 thread-safe하지 않습니다. 모든 MLX 추론은 단일 스케줄러 스레드에서 실행되고, RLock(`mlx_io_lock`)이 디스크 저장 등의 cross-thread 작업을 직렬화합니다. 이는 진정한 병렬성이 아닌 **시분할 협력적 동시성(time-sliced cooperative concurrency)**입니다. GPU가 merged batch 텐서를 단일 Metal kernel dispatch로 처리하므로 효과적입니다.

## 2.5 Cross-Phase Context Injection

멀티 페이즈 에이전트 워크플로우(협상, 토론, 심문 등)에서는 각 페이즈마다 context를 처음부터 재구성하는 것이 일반적입니다. 이 시스템은 KV cache를 **working memory**로 취급합니다:

- Phase 1: 초기 프롬프트 처리 후 캐시 저장
- Phase 2: Phase 1의 캐시를 로딩하고, prefix가 일치하는지 확인 후 EXTEND 매칭으로 새 context만 추가
- Phase N: 모든 이전 페이즈의 캐시가 누적

→ **프롬프트 템플릿이 append-only 구조를 따르므로, 캐시된 prefix가 항상 일치합니다.**

## 2.6 Architectural Coverage: ModelCacheSpec 추상화

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_4.png)

위 다이어그램은 두 가지 서로 다른 아키텍처(Gemma 3 12B의 GQA + Sliding Window, DeepSeek-V2-Lite 16B의 MLA + MoE)가 **ModelCacheSpec** 추상화를 통해 동일한 Block Pool, Q4 Pipeline, BatchQuantizedKVCache를 공유하는 구조를 보여줍니다.

- **Gemma 3 12B:** 48개 attention layer 중 8개는 global GQA, 40개는 sliding window(1024 토큰). GQA는 K=V=256(대칭)이며 8개 KV head를 16개 query head로 매핑합니다.
- **DeepSeek-V2-Lite 16B:** 27개 layer 모두 global attention. MLA(Multi-Latent Attention)로 K=192, V=128(비대칭)의 저차원 latent 표현을 사용합니다. MoE 라우팅은 중간 텐서를 생성하여 더 큰 메모리 예산(4096MB vs Gemma 2048MB)이 필요합니다.
- **Llama 3.1 8B:** 32개 layer, 표준 GQA(K=V=128).

> 모든 모델 차이는 **ModelCacheSpec 경계 아래**에서 처리됩니다. Block Pool, Q4 Pipeline, BatchQuantizedKVCache는 완전히 model-agnostic합니다.

# 3. Evaluation

---

## 3.1 실험 설정

- **하드웨어:** Apple MacBook Pro M4 Pro (24GB unified LPDDR5X, 273GB/s bandwidth)
- **모델:** Gemma 3 12B, DeepSeek-Coder-V2-Lite 16B, Llama 3.1 8B (모두 Q4 weights)
- **방법론:** 각 구성 6회 측정, 중앙값 보고. Temperature 0.0 (greedy decoding), 출력 64토큰 고정
- **캐시 상태:** Cold (캐시 없음, full prefill) / Warm (디스크에서 reload) / Hot (메모리에 상주)

## 3.2 TTFT Scaling

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_2.png)

위 그래프는 세 모델의 TTFT를 세 가지 캐시 상태(Cold/Warm/Hot)별로 context 길이(1K~32K)에 따라 보여줍니다. Y축은 로그 스케일입니다.

**핵심 관찰:**

- **Cold TTFT(파란색 실선 계열):** context 길이에 따라 **선형적으로 증가**합니다. Gemma의 32K cold prefill은 172초(약 3분)에 달합니다. $O(n)$ 스케일링을 명확하게 보여줍니다.
- **Warm TTFT(주황색 계열):** 거의 **수평선**입니다. 디스크 I/O + 캐시 복원 비용이 지배적이며, cache 크기에 따라 느리게 증가합니다. Gemma warm은 1K~32K에서 475~1819ms 범위입니다.
- **Hot TTFT(빨간색 계열):** Warm과 유사하지만 디스크 I/O가 없어 약간 더 빠릅니다.
- **Speedup:** 16K context에서 Gemma warm은 cold 대비 **93×**, DeepSeek warm은 **45×**, Llama warm은 **111×** 빠릅니다. 32K에서 Gemma는 **136×**까지 달합니다.

| 모델 | Cache | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|---|---|
| Gemma 3 | Cold | 3964 | 7119 | 15736 | 35009 | 74219 | 172096 |
| | Warm | 475 | 495 | 577 | 626 | 795 | 1819 |
| | Hot | 683 | 709 | 719 | 807 | 934 | 1264 |
| DeepSeek | Cold | 1043 | 1737 | 3970 | 8296 | 19396 | 47315 |
| | Warm | 234 | 246 | 271 | 338 | 434 | 633 |
| | Hot | 345 | 368 | 366 | 386 | 490 | 624 |
| Llama 3.1 | Cold | 2500 | 4530 | 10235 | 21536 | 47629 | — |
| | Warm | 260 | 274 | 290 | 331 | 431 | — |
| | Hot | 412 | 420 | 429 | 458 | 526 | — |

> **흥미로운 현상:** 짧은 context(1K~8K)에서 Gemma와 Llama의 **Hot TTFT가 Warm보다 40~55% 느립니다.** Warm 경로는 `mx.load()` 한 번의 최적화된 순차 I/O를 타는 반면, Hot 경로는 block pool의 산재된 메모리 접근(per-layer hash lookup, validation, scattered access)을 수행합니다. 캐시 파일이 수 MB로 작을 때는 순차 디스크 I/O가 산재된 메모리 접근보다 빠릅니다. 32K(캐시 3GB 이상)에서는 디스크 I/O가 지배적이 되어 Hot이 이깁니다.

## 3.3 Batched Throughput

두 에이전트를 동시 서빙할 때의 시스템 처리량입니다.

- Cold batched throughput은 낮습니다 (양 에이전트가 prefill에 막혀있으므로).
- **Warm/Hot에서 시스템 TPS:** Gemma 22.6 (4K), DeepSeek 52.6 (4K), Llama 34.5 (4K)
- DeepSeek이 2~3× 빠른 이유: MoE 아키텍처가 64개 expert 중 6개만 활성화하여 decode 단계의 연산량이 적습니다.
- **Warm-to-Hot 차이는 미미합니다.** 디스크 reload 레이턴시가 생성 과정에 분산되어 지속 throughput에 영향을 주지 않습니다.

## 3.4 vllm-mlx와의 비교

Llama 3.1 8B 기준으로 vllm-mlx(FP16 prefix cache, 메모리 내)와 head-to-head 비교를 수행했습니다.

| Context | agent-memory (Q4 KV) ||| vllm-mlx (FP16 KV) ||
|---|---|---|---|---|---|
| | Cold | Warm | Restart | Cold | Prefix |
| 1K | 2,500 | 260 | 260 | 2,289 | 273 |
| 4K | 10,235 | 290 | 290 | 4,394 | 290 |
| 8K | 21,536 | 331 | 331 | 9,472 | 244 |
| 16K | 47,629 | 431 | 431 | 37,935 | 371 |

**핵심 발견:**

- **Cold prefill:** vllm-mlx가 ~2.3× 빠릅니다. Q4 양자화 자체의 오버헤드는 3% 미만이고, 나머지 ~5.6초 차이는 agent-memory의 오케스트레이션 레이어(블록 풀 할당, 에이전트 ID 조회, 스케줄러)입니다.
- **Warm cache = FP16 prefix 성능:** 4K에서 agent-memory warm(290ms)과 vllm-mlx prefix(290ms)가 동일합니다. Q4 경로(290ms)가 FP16 경로(348ms)보다 오히려 빠른데, SSD에서 **4× 작은 파일**을 읽는 것이 ~7GB/s SSD 속도에서 유리하기 때문입니다.
- **서버 재시작 후:** vllm-mlx의 FP16 prefix cache는 **휘발성**이므로 cold prefill로 돌아갑니다. agent-memory는 디스크 영속성으로 warm 상태를 유지합니다.
- **메모리 밀도:** FP16 KV cache는 16K context에서 ~7개 에이전트, Q4는 **~30개** 에이전트를 수용합니다. vllm-mlx는 8K 이상의 multi-context 벤치마크에서 FP16 메모리 고갈로 실패했습니다.

## 3.5 Ablation Analysis

| 컴포넌트 | 메트릭 | With | Without | 효과 |
|---|---|---|---|---|
| Persistence | TTFT (ms), Gemma 4K | 577 | 15736 | **27×** |
| Q4 vs FP16 | Agents at 8K | 12 | 3 | **4.0×** |
| Batching | SysTPS, Gemma 1K warm | 22.6 | 11.2* | **2.0×** |
| Cross-phase | TTFT (ms), Phase 5 | 1705 | 3292 | **1.9×** |

- **Persistence**가 가장 큰 단일 개선(27×): 재계산을 완전히 제거합니다.
- **Q4 양자화**는 속도보다 **용량**에 기여: 8K에서 FP16은 3개, Q4는 12개 에이전트.
- **Cross-phase injection:** Phase 1에서는 이득 없음(둘 다 cold-start). Phase 5에서 persistent cache가 4개 이전 페이즈에 걸쳐 성장하여 **1.9× 빠른 reload**.

## 3.6 Staggered Arrivals: Batched 추론의 실질적 이점

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_3.png)

위 그림은 실제 멀티 에이전트 워크플로우에서의 성능을 보여줍니다. 상단 테이블(Table 14)은 M4 Pro 10.2GB cache budget에서 FP16 vs Q4의 **에이전트 수용량 비교**이고, 하단 막대 그래프는 **엇갈린 요청 도착(Agent B가 Agent A 2초 후 도착)** 시 sequential vs batched 모드를 비교합니다.

**에이전트 수용량 (Table 14):**
- 4K context: FP16은 Gemma 6개, Q4는 **24개** (4×). Llama는 FP16 19개 → Q4 **70개** (3.7×)
- 16K context: FP16은 Gemma **1개**(사실상 멀티 에이전트 불가), Q4는 **6개**
- 32K context: FP16은 Gemma **0개**(수용 불가), Q4는 **3개**

**엇갈린 요청 도착 (Bar chart):**
- **파란색 막대(Total wall time):** Sequential ≈ Batched로 거의 동일합니다. 전체 작업 완료 시간은 차이 없습니다.
- **주황색 막대(Agent B의 체감 TTFT):** 핵심 차이가 여기 있습니다.
  - Sequential: Agent B는 A의 처리가 **완전히 끝날 때까지 대기**한 후 시작합니다. Gemma에서 B의 TTFT는 ~34초입니다.
  - Batched: Agent B는 A의 batch에 **2초 후 즉시 합류**하여 동시 처리됩니다. GPU bandwidth를 공유하므로 개별 TTFT는 약 2× 늘어나지만, 대기 시간이 사라집니다.
- 사용자 관점에서 중요한 것은 **"원하는 시점부터 첫 토큰까지"** 시간입니다. Gemma: 36.5초(seq) vs 36.1초(bat), DeepSeek: 8.9초 vs 8.8초로 **사실상 동일하거나 batched가 약간 우위**입니다.

→ **Context가 커질수록 sequential의 대기 시간은 quadratic하게 증가하지만, batched의 concurrent processing은 Agent B의 응답성을 유지합니다.**

## 3.7 Multi-Phase Cache Persistence

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_5.png)

위 다이어그램은 **5-phase Prisoner's Dilemma** 시나리오에서 각 에이전트의 캐시 상태 전이를 보여줍니다. 4개 에이전트(Warden, Marco, Danny + 일시적 Analyst)가 5개 페이즈를 거칩니다. 색상이 캐시 상태를 나타냅니다: **파란색 = Cold(full prefill)**, **주황색 = Warm(디스크 reload)**, **분홍색 = Hot(in-memory)**.

- **Warden:** Phase 1에서 cold start(~4초) → Phase 2에서 warm(캐시 디스크에서 reload) → Phase 4에서 hot(~0.7초, 이미 메모리에 상주)
- **Marco:** Phase 1에서 cold → Phase 3에서 warm(2개 이전 페이즈의 캐시 누적) → Phase 4~5에서 hot
- **Danny:** Phase 2에서 cold → Phase 3에서 warm → Phase 4~5에서 hot
- **Analyst:** Phase 5에서만 등장하여 cold start — **일시적 에이전트**는 캐시 누적 혜택을 받지 못합니다

→ **영구 에이전트(Warden, Marco, Danny)는 cross-phase injection으로 캐시를 누적하며, 후반 페이즈로 갈수록 reload가 빨라집니다.** Phase 5에서 persistent 모드는 cold 대비 **1.9× TTFT 감소**, 총 wall time은 Gemma 기준 **23% 감소**(72.9초 → 56.1초)를 달성합니다.

## 3.8 Multi-Agent Routing

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/agent_memory_below_the_prompt/image_6.png)

위 다이어그램은 **Wikipedia 기반 10-expert 라우팅 벤치마크**를 보여줍니다. 10개의 전문가 에이전트(Bayesian, CLT, Regression, Hypothesis, Markov 등)가 각각 통계학 주제의 Wikipedia 문서(2~4K 토큰)로 priming됩니다. 색상이 캐시 상태를 나타냅니다: **파란색 = Cold**, **주황색 = Warm**, **분홍색 = Hot**.

**3-Phase 프로토콜:**
- **Phase 1 (Priming):** 10개 expert 모두 cold prefill로 자신의 전문 분야 문서를 처리합니다. Gemma 평균 20.5초/expert, DeepSeek 5.1초/expert.
- **Phase 2 (Cross-topic queries):** 사용자 질문이 관련 있는 2~3개 warm/hot expert에 라우팅됩니다. priming에서 캐시가 이미 디스크에 저장되어 있으므로 재계산이 불필요합니다.
- **Phase 3 (Repeated queries):** 동일 expert에 반복 질문 시 hot cache를 활용합니다.

| Phase | Gemma TTFT | Gemma Quality | DeepSeek TTFT | DeepSeek Quality |
|---|---|---|---|---|
| 1: Priming (cold) | 20,514ms | 8/10 | 5,140ms | 3/10 |
| 2: Queries (warm) | 847ms | 8/10 | 396ms | 4/10 |
| 3: Repeated (hot) | 860ms | 3/3 | 424ms | 2/3 |

→ **Priming 후 warm cache 쿼리는 24.2× 빠릅니다 (Gemma: 20,514ms → 847ms).** DeepSeek의 낮은 quality 점수(3~4/10)는 캐싱 품질 문제가 아니라, 코드 특화 모델(DeepSeek-Coder)의 일반 지식 한계를 반영합니다. Quality 평가 기준은 키워드 매칭 및 최소 길이 등 **구조적 품질**이지 사실적 정확성이 아닙니다.

## 3.9 Q4 Cache Quality

Perplexity로 Q4 양자화의 품질 영향을 측정했습니다 (WikiText-2, 512토큰 window, 7,935토큰).

| 모델 | FP16 PPL | Q4 PPL | Δ PPL | Δ% |
|---|---|---|---|---|
| Gemma 3 12B | 14.40 | 14.30 | −0.10 | **−0.7** |
| DeepSeek-V2-Lite 16B | 6.26 | 6.45 | +0.19 | **+3.0** |
| Llama 3.1 8B | 4.28 | 4.40 | +0.12 | **+2.8** |

- **Gemma:** −0.7% 변화는 측정 노이즈 범위 내. GQA의 대칭적 K=V=256 head dimension이 8개 KV head에 걸쳐 양자화 노이즈를 분산시킵니다.
- **DeepSeek:** +3.0% 증가. MLA의 비대칭 저차원 latent 표현(K=192, V=128)은 양자화 오류를 흡수할 여유가 적습니다.
- **Llama:** +2.8%. 표준 GQA(K=V=128)로, Gemma보다 작은 head dimension이 약간 더 큰 열화를 유발합니다.

→ **KIVI, KVQuant 등의 선행 Q4 양자화 연구와 일관된 결과이며, 실용적으로 허용 가능한 수준입니다.**

# 4. Discussion

---

## 4.1 Agentic 시스템의 인프라 레이어

이 시스템은 AutoGen, CrewAI, LangGraph 같은 에이전트 프레임워크 **아래**에 위치하는 인프라 레이어입니다. 에이전트 프레임워크가 역할 배정, 순서 결정, 도구 사용을 관리한다면, 이 시스템은 **어떤 캐시를 hot으로 유지하고, 어떤 것을 디스크로 내리고, 언제 reload할지**를 관리합니다.

- **Virtual memory 비유:** Block Pool은 page table, SSD는 swap, 멀티 에이전트 인터리빙은 page fault를 숨기는 temporal slack을 제공합니다.
- **Latency hiding:** 5-agent round-robin에서 Agent A가 생성(1~3초)하는 동안 Agent B의 캐시가 디스크에서 로딩(~500ms)됩니다. I/O가 decode 뒤에 완전히 숨겨집니다.

## 4.2 Persistent Cache vs RAG vs Message Passing

| | RAG | KV Persist | Msg Pass |
|---|---|---|---|
| Restore cost | $O(n)$ prefill | $O(n)$ I/O | $O(n)$ rebuild |
| Stores | Text chunks | Attn state | Structured msgs |
| Scope | External KB | Conv. history | Inter-agent |
| Hardware | Vector DB | SSD/RAM | Network |

→ **이 세 가지는 상호보완적입니다.** 하나의 에이전트가 외부 지식에 RAG를, 에이전트 간 조율에 message passing을, 자신의 대화 context 복원에 persistent KV cache를 동시에 사용할 수 있습니다.

## 4.3 Unified Memory가 중요한 이유

KV cache persistence의 실용성은 **스토리지 대 메모리 대역폭 비율**에 달려있습니다.

- **UMA 디바이스 (M4 Pro):** SSD ~7GB/s → 273GB/s 메모리 = SSD-to-UMA 비율 ~2.6%. SSD에서 로딩된 데이터가 즉시 full memory bandwidth로 접근 가능합니다.
- **Discrete GPU (RTX 5090):** VRAM에서 host RAM으로 spill 시 PCIe ~64GB/s → 1,792GB/s HBM = ~3.5% 비율. 비율은 비슷하지만, UMA는 로딩 후 즉시 full bandwidth 접근인 반면, **discrete GPU는 cache miss마다 PCIe 병목을 반복**합니다.

→ **이 시스템은 Apple Silicon 같은 UMA 아키텍처에서 가장 빛납니다.** Discrete GPU에서는 PCIe 왕복 비용이 반복되어 이점이 줄어듭니다.

## 4.4 Limitations

- **Single device:** 현재 모든 에이전트가 하나의 디바이스를 공유합니다. 멀티 디바이스 확장은 향후 연구.
- **Model-specific caches:** Gemma 3의 캐시를 다른 모델이나 다른 양자화로 재사용할 수 없습니다.
- **Fixed output length:** 모든 측정이 64토큰 출력 기준. 512토큰 출력 시 TTFT 절감의 상대적 비중은 줄어들지만 절대적 절감량은 동일합니다.
- **Q4 품질:** 512토큰 평가 윈도우로 측정했으며, Gemma의 sliding-window attention(1024)을 완전히 활용하지 못합니다. 더 긴 context 평가와 per-layer sensitivity 분석은 향후 연구.

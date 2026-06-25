# 1. Introduction

---

### 1. 배경 및 문제 제기

프런티어 LLM들이 빠르게 발전하면서, 흥미로운 현상이 함께 나타났습니다. **모델마다 잘하는 영역이 점점 갈라지고 있다**는 것입니다.

- 도메인 수준: GPT 계열은 수학·물리 추론, Opus 계열은 소프트웨어 엔지니어링·사이버보안에 강점
- 도메인 **내부**에서도 갈림: 경쟁 코딩에서 Gemini-3.1-Pro는 알려진 알고리즘을 직접 구현하는 데 능하고, GPT 계열은 여러 알고리즘 아이디어를 조합해 가장 어려운 문제를 푸는 계획 수립에 강함

여기에 더해, 최근 성능 향상은 모델 자체뿐 아니라 **agentic scaffold**(도구 사용, 환경 피드백, 메모리 관리 등으로 모델을 감싸는 구조)에서도 크게 나왔습니다. 즉 "능력"은 모델만의 속성이 아니라 **모델이 작동하는 scaffold의 속성**이기도 합니다.

> **핵심 질문:** 다음 frontier는 더 큰 단일 모델 하나가 아니라, 여러 모델의 상보적 강점을 *식별·결합·증폭*하는 시스템에서 오는 것 아닐까?

### 2. 제안: Sakana Fugu

**Sakana Fugu는 프런티어 LLM 에이전트 팀을 지휘(orchestrate)하도록 학습된 "오케스트레이터 모델" 패밀리**입니다. Fugu 자체가 언어 모델이며, 사용자 질의를 이해하고 그에 맞는 agentic scaffold를 **동적으로** 설계합니다.

핵심은, 사용자는 **단일 모델을 호출하듯** Fugu를 쓰지만 내부에서는 여러 전문 에이전트로 라우팅·위임·조정이 일어난다는 점입니다. 두 가지 변형을 공개했습니다.

| | **Fugu** | **Fugu-Ultra** |
|---|---|---|
| 목표 | 성능 ↔ 지연시간 균형 | 절대 답변 품질 최우선 |
| 동작 | 입력당 **단일 worker** 선택 | 입력당 **여러 에이전트 워크플로우** 구성 |
| 지연시간 | 프런티어 모델 직접 호출 수준 | 추가 지연 감수 |
| 기반 | Trinity (Xu et al., 2025) | Conductor (Nielsen et al., 2025) |

> 이는 모델 머징(model merging)의 **행동(behavioral) 수준 macro 버전**으로 볼 수 있습니다. 가중치를 평균내거나 레이어를 꿰매는 대신, 프런티어 모델을 **블랙박스 에이전트**로 두고 "어떻게 라우팅·조정·검증·종합할지"를 학습합니다. → 파라미터 접근이 필요 없으니 **closed-source 프런티어 모델까지** 그대로 조합할 수 있습니다.

### 3. 핵심 결과

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_1.png)

Figure 1은 Fugu/Fugu-Ultra를 프런티어 모델들과 8개 벤치마크에서 비교한 것입니다. **진한 빨강이 Fugu-Ultra, 밝은 빨강이 Fugu, 회색이 baseline**(각 provider 보고치)입니다.

- 코딩·터미널(Terminal Bench 2.1, LiveCodeBench), 과학 추론(GPQA-D), 차트 이해(CharXiv) 등 대부분에서 **빨강 막대가 회색을 앞서거나 최상위권**입니다.
- 특히 흥미로운 점은, Fugu의 worker pool에 **포함되지 않은** 비공개 모델(Fable 5, Mythos Preview)까지 일부 벤치마크에서 넘어섰다는 것입니다.
- 단, SWEBench Pro에서는 Fable 5(80.0)가 Fugu-Ultra(73.7)보다 높습니다. Fable 5는 agent pool에 없으므로 이는 오케스트레이션이 *모든* 비공개 모델을 능가한다는 뜻은 아닙니다.

> **핵심 메시지:** 모델 오케스트레이션은 "더 큰 모델을 훈련하는 것"과는 **별개의 새로운 scaling 축**이다. 학습 compute를 늘리지 않고도, 기존 프런티어 모델들을 똑똑하게 조합해 세대 업그레이드급 성능에 도달할 수 있다.

# 2. Fugu — 성능과 지연시간의 균형

---

Fugu는 응답 속도가 중요한 일상·대화형 워크로드를 겨냥한 변형입니다. 핵심 설계 철학은 **"오케스트레이터는 빠르고 가볍게 결정만 내린다"** 입니다.

> **⚠️ 가장 흔한 오해 먼저:** 아래에 나오는 hidden state·logit·lightweight head는 **closed worker 모델(Opus·Gemini·GPT)의 것이 아닙니다.** 그것들은 API로 호출되는 블랙박스일 뿐, 내부에 접근할 수 없습니다. hidden state/logit을 쓰는 주체는 **Fugu 자신** — 즉 Sakana가 따로 보유한 **별도의 오픈 LLM(orchestrator backbone)**입니다. backbone은 가중치를 가지고 있으니 자기 hidden state에 당연히 접근할 수 있죠. 정리하면 **"오픈 모델(지휘자)이 logit으로 결정 → closed 모델(연주자)이 실제 작업 수행"** 의 2층 구조입니다.

## 2.1 Parametrization: logit으로 결정하는 가벼운 head

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_2.png)

Figure 2가 Fugu의 모든 것입니다. 데이터 흐름을 따라가 보겠습니다.

1. 입력 텍스트("Write me code for binary search")가 backbone 언어 모델에 들어갑니다.
2. 모델의 어떤 토큰 위치(그림의 `<Head Input>`)에서 **hidden state $h \in \mathbb{R}^d$**를 꺼냅니다.
3. 기존 LM Head와 **병렬로** 작은 **Lightweight Head**가 이 $h$를 받아, 풀에 있는 $L$개 worker 각각에 대한 logit을 출력합니다. (오른쪽 빨간 노드들)
4. 가장 점수 높은 worker가 선택되어 질의를 통째로 위임받습니다.

여기서 **결정적으로 영리한 부분**이 두 가지입니다.

- **생성 텍스트가 아니라 logit을 쓴다.** 프롬프팅·실행은 선택된 worker가 다 하므로, Fugu는 worker 선택 결정만 내리면 됩니다. → 초반 토큰 위치에서 hidden state 한 번 계산 → head 적용 → 즉시 dispatch. **비싼 autoregressive 디코딩이 통째로 생략**됩니다. (이것이 낮은 지연시간의 핵심)
- **역할(role)을 할당하지 않는다.** 전신 모델 Trinity는 선택된 모델에 역할까지 부여했지만, Fugu는 항상 "worker"로만 dispatch합니다. → 조정 공간을 *모델 선택* 하나로 좁혀 오버헤드를 최소화.

추가로, backbone 전체를 fine-tuning하지 않고 **singular-value fine-tuning**(가중치 행렬을 분해해 특이값 스케일만 학습, 직교 성분은 고정 — 그림의 빨간 대각선)을 씁니다. → lightweight head와 합쳐도 **학습 파라미터가 극도로 작습니다.**

## 2.2 2단계 학습: SFT → 진화 전략

Fugu는 두 단계로 학습됩니다.

**① 단일 스텝 태스크 SFT.** 코딩·수학·추론 등 정답이 검증 가능한 질문 $q_i$를 대량 수집합니다. 각 worker 모델 $M_j$를 $q_i$에 대해 $n$번 돌려 보상을 측정하고, 모델별 평균 보상을 모아 점수 벡터를 만든 뒤, **softmax로 soft target 분포**로 바꿉니다.

$$p_i(j) = \frac{\exp(\bar{r}_{i,j}/\tau)}{\sum_{j'=1}^{K} \exp(\bar{r}_{i,j'}/\tau)}$$

이 분포를 정답으로 삼아 KL divergence를 최소화합니다.

$$\mathcal{L}_{\text{SFT}}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \mathbb{D}_{KL}\big(p_i(\cdot) \,\|\, \pi_\theta(\cdot \mid q_i)\big)$$

> 왜 hard label(1등 모델 하나) 대신 soft 분포일까요? **여러 worker가 비슷하게 유능할 때** 강제로 1등만 고르게 학습하면 결정이 불안정해집니다. 보상 크기 정보를 살린 soft 분포는 더 풍부한 신호를 주고, 선택을 robust하게 만듭니다.

**② End-to-end 태스크 진화 전략.** 단일 스텝 태스크는 깨끗하지만 실제 사용을 반영하지 못합니다. 그래서 Claude Code·Codex·OpenCode 등에서 **실제 멀티턴 trajectory**(저장소 컨텍스트, 반복 편집, 도구 호출, 실행 피드백)를 모아 end-to-end 태스크로 만듭니다.

- 이런 trajectory는 점수만으로는 안 보이는 차이를 드러냅니다. *어떤 모델은 고수준 추론은 강하나 도구 조작·파일 편집·피드백 반응에는 약하고, 어떤 모델은 벤치마크 점수는 평범해도 인터랙티브 harness 안에서 더 견고*합니다.
- 최적화는 **sep-CMA-ES**(진화 전략)로 terminal reward $R(\tau) \in \{0,1\}$의 기댓값을 직접 최대화합니다. SFT가 이미 파라미터를 좋은 영역에 놓았기 때문에, 진화 탐색은 라우팅 행동을 **미세하게 refine**하는 역할을 합니다.

→ **결과적으로 Fugu는 "고립된 답변 품질"이 아니라 "scaffold 안에서 도구·피드백과 함께 얼마나 잘 작동하는가"라는 실전적 worker 능력을 학습합니다.**

# 3. Fugu-Ultra — 성능 극대화

---

Fugu-Ultra는 가장 복잡한 워크로드에서 **절대 답변 품질**을 끌어내기 위한 변형으로, Conductor 프레임워크를 기반으로 합니다.

## 3.1 모델들의 오케스트라를 지휘하기

Conductor는 강화학습으로 학습된 언어 모델이 **자연어로 전체 agentic 워크플로우를 출력**하는 프레임워크입니다. 입력 태스크를 분할하고, 임의의 subtask를 할당하고, 타깃 커뮤니케이션 전략을 정의합니다.

각 워크플로우는 일련의 **step**으로 정의되며, 각 step은 세 가지를 명시합니다.

- 자연어 **subtask** 문자열
- 그 subtask를 수행할 **worker agent id**
- 이전 step들의 어떤 결과를 이 worker의 컨텍스트에 넣을지 정하는 **access list**

> 이 단순한 설계가 강력한 이유: best-of-N, 순차적 체인부터 임의의 **트리 구조 병렬 워크플로우**까지 자유롭게 표현할 수 있습니다. Conductor는 오케스트레이터 자신도 worker로 지정할 수 있어 조정 위상(topology)의 범위가 더 넓어집니다.

학습은 **GRPO**로 진행됩니다.

$$J(\theta) = \mathbb{E}_{q\sim D,\, \{o\}_1^G \sim \pi_\theta(\cdot|q)} \left[ \frac{1}{G}\sum_{i=1}^{G}\Big(\min\big(r_i A_i,\, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i\big)\Big) - \beta\,\mathbb{D}_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

보상 $r_i$는 두 단계 조건으로 정해집니다. **(1) 포맷 조건**: subtask·worker·access list가 파싱 불가능하면 0. **(2) 정답 조건**: 잘 구성된 워크플로우의 최종 출력이 정답과 일치하면 1, 아니면 0.5. (실제 학습은 KL 패널티 없이 진행)

→ 학습이 진행되면 **각 에이전트의 강점을 활용하는 문제 분해와 prompt-engineered subtask, 그리고 독립 작업과 에이전트 간 협업을 task에 맞게 섞는 커뮤니케이션 위상**이 자연스럽게 emergence합니다.

## 3.2 멀티 에이전트 함수 호출과 공유 메모리

멀티 에이전트 환경에서 **함수 호출(function calling)**은 독특한 메모리 문제를 일으킵니다. 단일 에이전트라면 메시지 transcript가 전체 컨텍스트를 담지만, Fugu-Ultra에서는 **아무 에이전트나 아무 때나** 함수를 호출할 수 있습니다. 따라서 시스템은 "어떤 에이전트가 어떤 호출을 했고, 그 에이전트가 워크플로우 어디에 있는지"를 추적해야 합니다.

이를 위해 두 가지 메커니즘이 절묘하게 균형을 이룹니다.

- **Intra-workflow 에이전트 격리.** 같은 워크플로우 안에서는 각 에이전트의 함수 호출 trajectory를 서로 격리합니다. 그렇지 않으면 *첫 에이전트가 전체 경로를 정해버리고 이후 에이전트들이 그 경로를 따라가 중복 기여만 하는* **orchestration collapse**가 발생합니다. 에이전트는 오직 access list를 통해서만 다른 에이전트의 작업을 봅니다.
- **Persistent shared memory.** 반대로 멀티턴 대화 전체에서 완전히 격리하면, 에이전트들이 같은 도구 호출을 반복해 이미 발견한 결과를 재발견하는 낭비가 생깁니다. 그래서 **이전 워크플로우의 도구 호출은 공유**하도록 inter-workflow 메모리를 허용합니다.

> 정리하면: **현재 워크플로우 안에서는 격리(독립적 탐색 보장), 대화 전체 수준에서는 공유(배경 컨텍스트 유지).** 이 두 축을 분리한 것이 Fugu-Ultra가 멀티턴 멀티 에이전트를 안정적으로 굴리는 핵심입니다.

학습 시 Ultra는 Gemini-3.1-Pro, Claude-Opus-4.8, GPT-5.5를 포함한 풀에서 **최대 5 step**의 워크플로우를 설계하도록 지시받으며, 멀티턴 태스크에서는 어떤 에이전트든 환경과 무제한 상호작용할 수 있습니다.

# 4. Benchmark 성능과 도메인 적응성

---

Fugu의 worker pool에는 Gemini-3.1-Pro, Claude-Opus-4.8, GPT-5.5 같은 SOTA 모델이 들어있습니다. 따라서 **"Fugu가 자기 worker들을 넘어서는가"**를 보려면, **바로 그 worker들과 동일한 최대 reasoning effort로** 비교해야 공정합니다.

## 4.1 모델 카드

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_3.png)

Table 1은 11개 벤치마크 종합 성적표입니다. **굵게가 1등, 밑줄이 2등**이고, 파란 음영 두 열이 Fugu-Ultra와 Fugu입니다.

- Fugu-Ultra는 SWE Bench Pro(73.7), Terminal Bench 2.1(82.1), LiveCodeBench(93.2), HLE(50.0), CharXiv(86.6), GPQA(95.5) 등 **대부분에서 1등**.
- 더 놀라운 건 **Fugu(단일 worker만 선택하는 빠른 모델)** 역시 SciCode(60.1), τ³ Banking(21.7), Long Context Reasoning(74.7), GPQA(95.5 공동 1등)에서 1등을 차지한다는 점.
- 즉 두 Fugu 모두, 자신이 호출하는 개별 SOTA 모델(Opus/Gemini/GPT)을 **상회**합니다. 오케스트레이션이 단순 라우팅을 넘어 실제 능력 증폭을 만든다는 증거입니다.

## 4.2 오케스트레이션은 "세대 업그레이드"와 맞먹는다

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_4.png)

Figure 3은 SWE-bench Pro의 해결률을 **시간축(2025년 11월~2026년 7월)**에 놓고, 각 provider의 세대별 모델(Opus 4.5→4.6→4.7→4.8, GPT-5.2→5.5, Gemini 3→3.1)이 어떻게 올라왔는지 그린 것입니다.

- 가장 오른쪽 위, 점선으로 연결된 **빨간 사각형이 Fugu-Ultra(73.7)**입니다. 최신 Opus 4.8(69.2)보다도 위에 있습니다.
- 핵심 해석: Opus가 4.7(64.3)→4.8(69.2)로 한 세대 오를 때 약 +5%p 올랐는데, **Fugu-Ultra는 그 다음 세대가 아직 나오지도 않았는데 그 자리를 미리 차지**하고 있습니다.

> → **즉 "오케스트레이션으로 얻는 이득"이 "한 세대의 모델 훈련으로 얻는 이득"과 같은 크기**라는 것입니다. 새 모델을 훈련하지 않고도 다음 세대 성능에 접근한 셈입니다.

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_5.png)

Figure 4는 더 나아가, 공개되지 않은 **Mythos Preview·Fable 5 모델 클래스**와 비교합니다. GPQA-Diamond(95.5), CharXiv Reasoning(86.6), Terminal Bench 2.1(82.1) 세 과학·추론 벤치마크에서 **빨간 막대(Fugu)가 비공개 최상위 모델까지 넘어섭니다.** Fugu는 GPT의 수학 강점을 알아보고 계산이 필요한 곳에 GPT를 타깃 투입하는 식으로 이 격차를 만들어냅니다.

## 4.3 도메인 적응성 — 라우팅이 실제로 "전문성"을 따라간다

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_6.png)

Figure 5는 태스크별로 어떤 worker가 얼마나 선택됐는지(분포)를 보여줍니다. 세 그룹(HLE, Terminal Bench, GPQA-Diamond)에서 색은 에이전트(파랑 Gemini, 주황 Opus, 검정 GPT), **꽉 찬 막대는 Fugu-Ultra, 빗금 막대는 Fugu**입니다.

- **Terminal Bench**: GPT-5.5(검정)가 SOTA인 도메인 → 분포가 GPT로 크게 쏠림(Fugu 0.86, Ultra 0.64).
- **GPQA-Diamond**: Gemini가 선두인 도메인 → 분포가 Gemini(파랑)로 집중(Ultra 0.56).
- **HLE**: 본질적으로 다학제적 → 세 에이전트에 **고르게 분산**. 세부적으로는 수학 문제는 GPT, 화학·생물은 Gemini로 라우팅됩니다.

> → **라우팅 분포가 우리가 아는 모델별 SOTA 사전(prior)과 일치**합니다. Fugu가 단순히 한 모델로 쏠리는 게 아니라, **도메인의 성격에 따라 전략 자체를 바꾼다**는 것이 적응성(adaptivity)의 직접 증거입니다.

# 5. 벤치마크를 넘어선 실전 능력

---

집계 점수만으로는 안 보이는 **실전 agentic 행동**을 보려고, 저자들은 정성적 end-to-end 태스크를 만들었습니다. 비교 대상(Gemini 3.1 Pro, Opus 4.8, GPT 5.5)은 **Model A/B/C로 익명화**되고 예시마다 매핑을 바꿔, 브랜드가 아닌 *행동 차이*에 집중하게 했습니다.

## 5.1 자율 ML 연구 최적화 (AutoResearch)

작은 GPT 학습 파이프라인을 에이전트가 **반복적으로 수정→실행→개선**하며 검증 BPB(bits-per-byte, 낮을수록 좋음)를 줄이는 태스크입니다. 단일 H100에서 시드당 **123회 자율 실험(~14시간)**, 3 시드.

![image_7](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_7.png)

Figure 6의 가로축은 자율 실험 횟수, 세로축은 검증 BPB입니다. **굵은 빨강이 Fugu-Ultra 평균, 빨강 점선이 best run**, 파랑/초록/노랑이 Model A/B/C입니다. 빨간 곡선 위의 동그라미 번호들은 Fugu-Ultra가 발견한 주요 변경점입니다(batch size $2^{19}\!\to\!2^{18}$, depth 8→9, learning rate 조정 등).

- 최종적으로 Fugu-Ultra가 **최저 평균 BPB(0.9774)**로 Model C(0.9781)·B(0.9793)·A(0.9822)를 모두 앞섭니다.
- 절대값 차이는 작지만(이미 고도로 최적화된 파이프라인이라 당연), **평균과 best-run 양쪽 모두에서 일관**되며 전 과정에 걸쳐 유지됩니다.

![image_8](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_8.png)

Table 2가 최종 수치입니다. 두 가지가 특히 시사적입니다. **(1)** Fugu-Ultra는 초반엔 비슷하다가 **중반 이후 치고 나갑니다** → 오케스트레이션은 탐색이 거친 설정 변경에서 미세한 optimizer/schedule 튜닝으로 넘어갈 때 가장 가치가 큽니다. **(2)** Model B는 best seed는 좋지만 **시드 간 분산이 큽니다**(±0.0025). Fugu-Ultra는 peak와 일관성을 **동시에** 개선합니다.

## 5.2 고전 일본어 가나 letter 읽기 순서 복원

학습 데이터가 **존재하지 않는** 영역의 사례입니다. 고전 가나 편지(kana-shōsoku)는 **chirashigaki(흩뿌려 쓰기)** 스타일로, 문자가 페이지에 다양한 크기·위치로 산재해 있어 훈련된 독자조차 읽기 순서를 잡기 어렵습니다. 공개 데이터셋이 없어 저자들이 전문가 주석 25페이지를 직접 만들었습니다.

![image_9](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_9.png)

Figure 7은 한 페이지 결과입니다. **위: 원본 편지, 가운데: Fugu-Ultra(NED 0.80), 아래: frontier baseline(NED 0.24)**. 예측한 읽기 경로(**빨강**)가 전문가 정답(**초록**) 위에 겹쳐져 있습니다.

- Fugu-Ultra의 빨강 경로는 초록(전문가 traversal)을 **충실히 따라갑니다**.
- baseline의 경로는 산재한 문자들 사이를 어지럽게 가로질러 정답과 어긋납니다.
- 핵심 아이디어: Fugu는 모델을 *훈련*하는 게 아니라, **읽기 순서 예측 함수(코드)를 직접 작성**하고 test-time scaling(beam search)으로 개선합니다. 데이터가 없으니 "암묵적 규칙 → 작동하는 알고리즘"으로 번역하는 능력을 시험하는 것입니다.

![image_10](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_10.png)

Table 3은 25페이지 전체 평균 NED(높을수록 좋음)입니다. **모든 모델에 같은 beam search를 적용**했으므로 결과는 곧 *탐색을 모는 모델의 힘*을 반영합니다. Fugu-Ultra가 0.776으로 최고, 다음이 Model A(0.642), Fugu(0.473), Model B(0.449), seed heuristic(0.116) 순입니다. → **탐색을 고정했을 때 천장을 정하는 것은 모델이며, 그 엔진으로서 Fugu-Ultra가 가장 강합니다.**

## 5.3 CAD 생성 — 작동하는 기계 메커니즘

텍스트 지시로 **mechanical iris**(카메라 조리개 메커니즘)를 생성하는 태스크입니다. 단순한 원형 부품이 아니라 여러 blade, outer pin, 회전 ring, 그리고 열고 닫을 때도 일관된 중앙 개구부가 필요합니다.

![image_11](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_11.png)

Figure 8은 각 모델의 CAD view와 단순화 view를 open/closed 상태로 비교합니다. **파랑은 Fugu-Ultra의 올바른 개폐 영역, 빨강은 다른 모델들의 불완전한 커버리지/약한 링크**를 표시합니다.

- **Fugu-Ultra**: 각 blade가 outer pin을 중심으로 회전하며 중앙 개구부를 매끄럽게 넓히고 좁힙니다.
- **다른 모델들**: blade가 중심을 다 덮지 못하거나, outer link가 기계적으로 약하거나, 개구부가 충분히 닫히지 않습니다.

## 5.4 그 외 — 큐브 솔버, 블라인드 체스, 온라인 트레이딩 (Appendix)

**Rubik's Cube 솔버 합성(Table 4):** 표준 라이브러리만으로 큐브 솔버를 *한 번에* 작성. Fugu 두 모델은 300/300 완전 해결, 프런티어 baseline 3개 중 **2개는 단 한 큐브도 풀기 전에 크래시**. Fugu-Ultra는 평균 19.72 HTM(최적=20)로 가장 짧은 해를 내고, Fugu는 약 1수를 더 쓰는 대신 **35배 빠른**(큐브당 1.9초) 솔버를 만듭니다. → *"실제로 돌아가는 코드를 만드는 신뢰성"*이 바로 baseline이 무너지는 지점.

![image_12](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_12.png)

**Blindfold Chess(Figure 9):** 보드를 ASCII/FEN으로 절대 보여주지 않고, 매 턴 상대의 마지막 수만 좌표 표기로 주며 **머릿속으로만** 전체 국면을 추적하게 합니다. 각 열은 Fugu vs Model A/B/C, 그리고 전문가급 Stockfish 18(~2100 Elo)와의 대국입니다. 위는 개시 국면, 가운데는 최종 체크메이트, 아래는 Fugu 승률 곡선입니다. **Fugu가 네 판 모두 승리**하며, 자체 blunder/실수 없이 상대보다 정확하게(낮은 centipawn loss) 둡니다. Model A전에서는 약간 불리하게 시작해 **역전승**하는 곡선이 인상적입니다.

![image_13](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/sakana_fugu/image_13.png)

**Online Sequential Trading(Figure 10):** 미래 정보 없이 주당 한 번 매수/보유/매도와 사이징(25/50/100%)을 결정하는 50주 온라인 트레이딩. $10,000에서 시작해 Fugu-Ultra(파랑)는 **$11,943.22 ± $633.86 (+19.43%)**에 도달, 다른 프런티어 모델들은 모두 +15% 미만에 그칩니다. 노이즈 섞인 시장 신호를 **잘 타이밍 잡힌, 적절히 사이징된 매매**로 번역하는 능력 차이를 보여줍니다.

# 6. 최적 전략과 위상 (정성적 인사이트)

---

Fugu-Ultra의 가장 큰 강점은 **자연어로 표현 가능한 어떤 조정 위상이든 질문별로 만들어낸다**는 점입니다. 실제로 학습 결과 다음과 같은 전략들이 자발적으로 나타났습니다.

**① Debate & Aggregation (토론과 종합).** 지식 집약 도메인에서 트리 구조가 나타납니다.
- HLE의 한 게임 메커니즘 문제: Gemini를 **트리 꼭대기(aggregator)**에 두고, leaf에 Gemini와 GPT를 독립 배치. 두 leaf가 각각 부분 정답(하나는 방어 수치 오류, 하나는 규칙 적용 오류)을 냈지만, **aggregator Gemini가 양쪽의 옳은 부분만 식별해 완전한 정답으로 종합**.
- 반대로 수학 문제(Calabi–Yau 불변량)에서는 **GPT를 aggregator**로 두고 Gemini·Opus를 leaf에 배치, GPT가 둘의 정수 불일치를 spectral 함수 재유도로 해결.

> → 핵심은 **aggregator 역할을 질문마다 바꾼다**는 것. 기존 멀티 에이전트 시스템은 항상 고정된 모델이 최종 종합을 맡아, 그 모델의 전문 영역 밖에서는 병목이 됩니다. Fugu-Ultra의 동적 aggregator는 이 한계를 우회합니다.

**② Build & Debug (빌드와 디버그).** 멀티턴 코딩에서 자주 관찰된 패턴: **GPT를 builder로, Opus를 결정적 순간의 검증·취약점 노출에 투입.**
- Terminal Bench의 PyPI 서버 구축: GPT가 먼저 빌드 → Opus가 risk를 enumerate(plain http.server 오용, 취약한 wheel 빌드, Debian-slim 환경 관리 실패) → GPT의 reachability 체크가 잘못된 orphaned 프로세스에서 왔음을 발견 → 피드백 후 GPT가 성공적으로 완성.
- SWE Bench Pro의 OTP 버그: Opus가 서버 측을 깊이 파다 막다른 길 → GPT가 *클린 슬레이트*로 재검토해 "서버가 아니라 client-side concurrency 버그"임을 발견 → Opus가 방향을 바꿔 shared ContextReader 도입으로 해결.

**③ 전문가 소환 (Bringing in a specialist).** 특정 지식이 필요할 때 추가 모델을 선택 투입.
- FEAL 차분 암호분석: Opus의 사이버보안 전문성으로 chosen-plaintext attack 초안 → 이후 GPT를 **수학 전문가로 명시 지정**해 cipher를 bit 단위로 추적, 공격 성공에 필요한 differential constant를 재유도.

> → 세 전략 모두 공통적으로 **per-question 단위의 미세 적응성**을 보여줍니다. cybersecurity·engineering·math 같은 **교차 도메인 전문성을 한 문제 안에서 결합**하는 것이 Fugu-Ultra가 개별 모델의 한계를 넘는 방식입니다.

# 7. 자주 묻는 질문 (FAQ)

---

읽다 보면 자연스레 떠오르는 질문들을 정리했습니다.

**Q1. 결국 상용 LLM API 오케스트레이션을 대신 해주는 시스템인가요?**
네. 사용자는 단일 모델을 부르듯 Fugu를 호출하고, 내부에서 closed 프런티어 모델들로 라우팅·조정이 일어납니다. 차이는 그 "지휘자"가 규칙 기반이 아니라 **학습된 모델**이라는 점입니다.

**Q2. closed 모델인데 어떻게 logit/hidden state를 쓰나요?**
worker(closed)의 것이 아니라 **Fugu 자신의 오픈 backbone**의 것입니다 (2장 상단 경고 참고). worker는 끝까지 블랙박스 API로만 쓰입니다.

**Q3. Fugu-Ultra는 "누가 무엇을 할지"를 어떤 기준으로 정하나요?**
명시적 규칙이 없습니다. "GPT=수학, Opus=디버깅" 같은 배정을 주입한 게 아니라, **최종 정답을 맞히면 보상**을 주는 GRPO 학습 과정에서 그런 배정이 **emergent하게** 떠오른 것입니다. 즉 "이 subtask를 이 worker에 주면 정답 확률이 높더라"를 정책이 내재화한 결과입니다. (Fugu 쪽은 다릅니다 — 각 모델을 실제로 돌려 측정한 성능 통계를 soft label로 학습합니다. 2.2절 참고)

**Q4. 모델 풀에 새 모델이 추가되거나 기존 모델이 업그레이드되면 다시 학습해야 하나요?**
사실상 **그렇습니다.** 이 점은 논문이 "composability(갈아끼우기)"를 장점으로 내세우면서도 비용은 두루뭉술하게 넘긴 부분입니다.
- **Fugu:** lightweight head가 풀 크기 $L$에 맞춰 $L$개 logit을 출력하므로, 모델 추가 시 head 출력 차원이 바뀝니다. 다만 비용은 **"거대 모델 재학습"이 아니라 "새 모델을 코퍼스에 돌려 라벨 재측정 + 작은 head 재학습"** 수준이라 상대적으로 쌉니다. 모델이 업그레이드돼 성능 프로파일이 바뀐 경우도, 라벨을 갱신하지 않으면 옛 성능 기준으로 라우팅하는 stale 상태가 됩니다.
- **Fugu-Ultra:** worker가 정수 id로 워크플로우에 박혀 있고 정책이 특정 풀 스냅샷에 맞춰 최적화돼 있어, 풀이 의미 있게 바뀌면 **추가 GRPO**가 필요합니다 (Fugu보다 비쌈).

> 즉 "한 번 학습하면 영원히 자동 적응"이 아니라, **"풀 갱신 시 재최적화 비용이 모델 훈련보다 훨씬 싸다"**가 논문이 실제로 지지하는 주장입니다. 재최적화의 정확한 비용·빈도·자동화 방법은 이 report의 빈칸으로 남아 있습니다.

---

## 한 줄 요약

> **Sakana Fugu는 "더 큰 모델"이 아니라 "더 똑똑한 지휘"로 frontier를 넘는다.** 프런티어 LLM들을 블랙박스 worker로 두고, 질문마다 누구를 부르고·어떻게 소통시키고·어떻게 종합할지를 학습한 오케스트레이터다. Fugu는 단일 worker를 빠르게 고르고, Fugu-Ultra는 멀티 에이전트 트리 워크플로우를 짠다. 그 결과는 개별 SOTA 모델은 물론, 비공개 차세대 모델 클래스까지 넘어서는 **세대 업그레이드급 성능 — 학습 compute 추가 없이.**

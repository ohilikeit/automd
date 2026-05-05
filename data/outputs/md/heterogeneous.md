# 1. Introduction

---

### 1. 배경 및 문제 제기: 언어로 모든 걸 풀기엔 과학이 너무 다양하다

- LLM 기반 agentic AI 시스템은 **자연어를 universal interface**로 가정합니다. 하지만 과학 분야의 데이터는 자연어가 아닙니다.
  - 단백질 시퀀스, 시계열 신호, 분자 구조, 표 형식의 임상 데이터, 수식, 위성 영상…
  - 이런 modality를 굳이 자연어로 직렬화(serialize)하면, 토큰 비용은 폭증하고 정확도는 오히려 떨어집니다.

- **그래서 도메인 전용 Foundation Model(FM)들이 따로 발전해왔습니다.**
  - 예: **Chronos**(범용 시계열 FM), **TabPFN**(transformer 기반 in-context tabular 예측 FM), 분자/단백질/지구관측용 FM 등.
  - 이들은 자기 도메인에서 LLM보다 **압도적으로 잘 합니다.** 그러나 치명적 단점이 있습니다.

    → **이 FM들은 자연어 인터페이스가 없습니다.** Agent 시스템 안으로 그대로 끌어들일 방법이 없는 겁니다.

- **연구 질문 (논문이 던지는 한 문장):**

> *"Can heterogeneous foundation models collaborate within agentic systems?"*
> 이질적 foundation model들이 하나의 agentic system 안에서 함께 일할 수 있는가?

### 2. 제안: Eywa — 영화 Avatar의 *Tsaheylu*에서 영감을 받은 프레임워크

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_2.png)

이 그림이 논문 전체의 비유 구조를 한 장에 담고 있습니다. 위쪽은 **영화 Avatar의 Pandora 행성**, 아래쪽은 그것을 **Hugging Face 생태계**로 옮긴 것입니다.

- **Pandora의 Tsaheylu (촐나)**: 나비족(Na'vi)은 머리카락 끝에 신경 다발이 있어, 다른 종(예: 마운틴 밴시, 다이어호스)과 직접 신경적으로 연결됩니다. 언어가 통하지 않아도 **신경 통신으로 협업**합니다. 그리고 행성 전체는 **"All Mother(Eywa)"**가 균형을 잡습니다.
- **이를 AI 생태계로 옮기면:**
  - **Na'vi 전사** = LLM agent (일반화된 추론을 함)
  - **Mountain Banshee** = 도메인 특화 FM (Chronos, TabPFN 등 — 강력한 전문 능력)
  - **Tsaheylu (촐나)** = LLM과 FM 사이의 **양방향 구조화된 통신 인터페이스**
  - **All Mother** = 전체를 조율하는 planner

→ **언어를 universal interface로 강제하지 말고, 모달리티-네이티브한 협업을 허용하자.** 이것이 Eywa의 한 줄 요약입니다.

이 프레임워크는 **3단계 구성**으로 진화합니다:

1. **EywaAgent**: 단일 도메인 FM에 LLM을 *bond*시킨 단일 agent (Banshee와 결합한 Na'vi 전사 한 명)
2. **EywaMAS**: 기존 multi-agent system의 LLM agent 자리에 EywaAgent를 plug-and-play로 끼워넣은 시스템
3. **EywaOrchestra**: planner가 **태스크별로 동적으로** LLM agent와 EywaAgent를 골라 조합하는 오케스트레이션

### 3. 핵심 결과 (한눈에)

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_1.png)

위 그림은 논문의 결과를 한 장에 응축했습니다. **왼쪽**은 baseline 대비 utility-cost trade-off를, **오른쪽**은 카테고리별 결과(Physical / Life / Social science)를 보여줍니다.

- **왼쪽 위 산점도 (Pareto frontier):** 가로축이 토큰 소비, 세로축이 utility(높을수록 좋음). EywaAgent / EywaMAS / EywaOrchestra(별표)가 **Pareto frontier 위쪽**에 위치 — 즉 **더 적은 토큰으로 더 높은 성능**입니다.
- **오른쪽 위 막대(주황색):** Single-LLM-Agent 대비 EywaAgent(Ours). 도메인별로 **Utility +7%, Token 최대 -34%, Time 최대 -14%** — 평균 token -30%, time -10% 절감.
- **오른쪽 아래 막대(보라색):** Multi-LLM-Agents 대비 EywaMAS. **Utility +3.5~5.9%, Token -12~18%, Time -4~23%** 수준.
- **왼쪽 아래 레이더 차트:** 9개 sub-domain(Material, Energy, Space, Biology, Clinic, Drug, Economy, Business, Infrastructure)에서 **모두 baseline 대비 우위** — 즉 특정 도메인에 편중되지 않은 일반적인 개선.

→ **추가 학습 없이, 단지 "LLM ↔ FM 인터페이스를 MCP로 깔끔하게 만든 것"만으로 utility ↑, token ↓, time ↓ 세 마리 토끼를 동시에 잡았다는 것이 핵심 메시지입니다.**

# 2. EywaAgent: FM과 LLM을 신경 결합하는 법

---

## 2.1 Tsaheylu 인터페이스 — 두 개의 함수로 정의하기

Eywa의 가장 작은 단위인 **EywaAgent**는 본질적으로 LLM과 도메인 FM을 짝지은 **"bonded duo"**입니다. 그런데 어떻게 짝지을까요? 논문은 이를 **두 개의 함수로 형식화**합니다.

각 도메인 $k$에 대해 인터페이스 쌍 $(\phi_k, \psi_k)$:

- $\phi_k : \mathcal{S} \to \mathcal{U}_k$: **query compiler**. LLM의 task state를 FM이 먹을 수 있는 **구조화된 호출**로 번역. (= "촐나로 의도를 전달")
- $\psi_k : \mathcal{O}_k \to \mathcal{Z}_k$: **response adapter**. FM 출력을 LLM 추론에 다시 연결할 수 있는 형태로 변환. (= "촐나로 결과를 받아와 해석")

전체 파이프라인:

$$\tau \xrightarrow{A_{LLM}} s \xrightarrow{\phi_k} u_k \xrightarrow{F_k} o_k \xrightarrow{\psi_k} z_k \xrightarrow{A_{LLM}} \hat{y}$$

읽는 법: 자연어 task → LLM이 해석해서 state로 → query compiler가 FM 호출 인자로 → FM 실행 → 결과를 response adapter가 추론 가능한 형태로 → LLM이 종합해서 최종 답.

> **핵심 직관:** 언어 모델은 "어떤 FM을, 어떤 인자로, 언제 부를지"만 결정하면 됩니다. 실제 도메인 계산은 FM에게 전적으로 위임. 마치 **장군은 지휘만, 전문병은 자기 일에 집중**하는 분업과 같습니다.

## 2.2 MCP로 구현하기 — Tsaheylu의 실체

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_3.png)

이 그림은 EywaAgent의 실제 구현 모습을 보여줍니다.

- **왼쪽(빨간 박스):** 기존 방식의 한계입니다. FM에게만 물으면 자연어를 못 알아듣고 일반 추론도 못 합니다 ❌. LLM에게만 물으면 전문 도메인 지식이 부족해서 suboptimal ❌.
- **오른쪽(녹색 박스):** EywaAgent의 해법.
  - **사용자 입력**(자연어) → LLM이 받음
  - LLM이 task를 해석하고 **query compiler($\phi_k$)** 를 통해 FM에 **structured invocation** 전송
  - **FM on Server**(Chronos, TabPFN 등)가 자기 영역에서 일을 처리하고 **response** 반환
  - **response adapter($\psi_k$)** 가 결과를 LLM이 해석할 수 있는 형태로 변환
  - LLM이 종합하여 자연어 출력

**구현은 Model Context Protocol (MCP) 위에서:**
- 각 FM은 독립 MCP backend로 띄워짐 (FastMCP 서버 + LangChain agents)
- query compiler는 **structured tool call** (대상 리소스, 데이터셋 ID, 호출 파라미터 등을 명세)
- LLM이 "도구를 부를지, 그냥 추론으로 끝낼지" 동적으로 결정

> ✅ FM은 LLM이 설정해주는 인자로 자연어 agent처럼 통신하고
> ✅ 전문 계산은 FM에 완전히 위임되어 도메인 정확도가 보장됨

## 2.3 왜 이게 이론적으로 더 나은가?

논문은 **Domain Advantage Assumption**(가정 1)을 세웁니다:

$$\mathbb{E}_\tau [\ell_k(F_k(x_k), y^\star)] < \inf_{A_{LLM}} \mathbb{E}_\tau [\ell_k(A_{LLM}(\text{serialize}(x_k)), y^\star)]$$

읽는 법: 도메인 $k$의 FM $F_k$가 **자연어로 직렬화한 입력**으로 추론하는 어떤 LLM보다 **strictly 낮은 loss**를 달성.

이 가정 아래 **Theorem 3**: EywaAgent의 최적 risk는 language-only agent의 최적 risk보다 **strictly 작다**.

→ **즉, FM이 자기 도메인에서 LLM보다 잘하기만 하면, EywaAgent는 항상 LLM-only agent보다 낫다는 것이 보장됩니다.** 직관적으로도 너무 당연한데, 이를 식으로 못박은 것이 Theorem 3의 의의입니다.

# 3. EywaMAS와 EywaOrchestra: 다중 에이전트로 확장

---

## 3.1 EywaMAS: 기존 시스템에 그냥 끼워넣기

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_4.png)

위 그림이 EywaMAS의 핵심 디자인을 보여줍니다.

- **위 박스(Existing MAS)**: 기존 multi-agent system은 LLM agent들로만 구성. Sequential, Looped, Hierarchical 같은 토폴로지가 있지만 모두 **언어 기반 통신**.
- **아래 박스(EywaMAS)**: 똑같은 토폴로지를 유지한 채, **일부 agent를 EywaAgent로 교체**. 추가로 **Heterogeneous** 토폴로지가 자연스럽게 생김.

→ **이것이 plug-and-play의 진짜 의미입니다.** 시스템 구조를 갈아엎지 않고도 도메인 FM의 능력을 multi-agent 시스템에 주입할 수 있습니다.

**예시:** Hierarchical MAS(planner + worker × N + summarizer)가 있다면, worker 일부만 EywaAgent로 바꾸면 끝. Planner와 summarizer는 그대로 LLM agent.

## 3.2 EywaOrchestra: planner가 동적으로 팀을 짠다

EywaMAS의 한계는 **토폴로지가 고정**이라는 것입니다. 하지만 태스크마다 최적 구조가 다릅니다.

- 단순 추론 태스크: single-agent EywaAgent로 충분
- 복잡한 다단계 분석: heterogeneous MAS가 유리
- Economy/Business 도메인: 사실 single-agent가 이미 잘 함

→ **태스크마다 시스템 구성 자체를 바꿔야 한다.** 그래서 등장하는 것이 **EywaOrchestra**.

**핵심 알고리즘 (Algorithm 1):**

```
입력: 태스크 τ = (q, x, y*, ℓ), 구성 공간 𝒞, conductor P
1. 시스템 구성 c ← P(q, x)            # conductor가 태스크 보고 구성 결정
2. c가 명세하는 heterogeneous agent system 인스턴스화
3. (q, x)에 대해 실행하고 결과 ŷ 반환
```

**Conductor가 결정하는 4가지:**
1. 각 agent의 역할/유형 (LLM agent or EywaAgent)
2. agent별 backbone LLM
3. EywaAgent의 도메인 FM
4. 전체 multi-agent 토폴로지

논문에서는 conductor 자체도 LLM으로 구현했습니다. 즉 **"메타 LLM이 태스크를 보고 팀을 짜는"** 구조입니다.

**이론적 정당성:** 어떤 고정 구성 $c$의 best risk $\mathcal{R}^*_{\text{fixed}}$ 보다, 태스크별 적응 구성의 oracle risk $\mathcal{R}_{\text{oracle}}$ 가 항상 더 낮습니다 ($\mathcal{R}_{\text{oracle}} \leq \mathcal{R}^*_{\text{fixed}}$). 다른 태스크가 다른 구성을 선호하는 경우엔 **strict** 부등호.

# 4. Experiments

---

## 4.1 EywaBench: 새로 만든 benchmark

기존 과학 benchmark의 문제: **(1)** task가 너무 좁거나, **(2)** 단일 도메인, **(3)** 단일 데이터 형식에 갇혀있음. 특히 **시계열과 tabular 데이터**가 거의 평가되지 않음.

**EywaBench의 구성:**
- **3개 대분류 × 3개 sub-domain = 9개 도메인**
  - Physical: Material / Energy / Space
  - Life: Biology / Clinic / Drug
  - Social: Economy / Business / Infrastructure
- **3가지 데이터 모달리티**: 자연어 / 시계열 / tabular
- **통합 utility score** $u \in [0, 1]$ — 모든 모달리티에서 비교 가능

**FM 두 개 사용:**
- **Chronos** — 범용 시계열 FM
- **TabPFN** — Transformer 기반 in-context tabular 예측 FM
- 둘 다 **자연어 인터페이스가 없음** (그래서 Eywa의 의미가 살아남)

**구현:** Tsaheylu는 LangChain + FastMCP로 구현. 각 FM은 **streamable HTTP**로 별도 MCP backend에서 서빙. CPU(13세대 i9-13900H, 64GB RAM)에서 측정.

## 4.2 메인 결과: 세 가지 차원에서의 우위

![image_7](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_7.png)

이 표는 논문의 핵심 실험 결과입니다. 각 셀에는 **Utility(↑), Time(↓, 초), Tokens(↓)** 세 metric이 있습니다.

**중요한 비교 포인트:**

**(a) Single-Agent 비교 (Single-LLM-Agent vs EywaAgent):**
- Overall Utility: **0.6154 → 0.6558** (+6.6%)
- Tokens: **4469 → 3137** (-30%)
- Time: 25.22 → 22.78 (-10%)
- → **단일 에이전트만으로도 거의 모든 metric이 개선.** 특히 Economy(0.7689→0.8048)와 Drug에서 두드러짐.

**(b) Multi-Agent 비교 (Refine/Debate/MoA/X-MAS vs EywaMAS):**
- EywaMAS는 **Multi-LLM 기반 baseline 대비 모든 도메인에서 best 또는 second-best** (볼드/언더라인).
- 특히 **Debate MAS와 동일 토폴로지를 쓰면서도 더 적은 토큰**을 사용함.
- → **언어 기반 heterogeneity(다른 LLM들 섞기)만으로는 부족하다. 모달리티-네이티브 heterogeneity가 진짜 차이를 만든다.**

**(c) Dynamic Orchestration (EywaOrchestra):**
- **고정 expert config 없이 conductor가 자동으로 팀 구성.**
- Utility는 EywaMAS에 근접(0.6746 vs 0.6761)하면서, **token은 8335 vs 11214로 25% 더 적게** 사용.
- 일부 sub-domain(Space, Drug, Business)에서는 EywaMAS도 추월.

> 💡 **핵심 인사이트: 모든 태스크에 무거운 multi-agent를 쓰면 낭비다.** Economy/Business에서는 single-agent EywaAgent가 이미 충분히 강력. Conductor가 태스크별로 가벼운/무거운 구성을 자동 선택하는 EywaOrchestra가 실용적으로 가장 매력적임.

## 4.3 Pareto Frontier로 보는 "공짜 점심"

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_5.png)

이 그래프는 **utility-cost trade-off**를 한 장에 압축합니다.

- **가로축**: 토큰 소비량 (낮을수록 좋음)
- **세로축**: Utility (높을수록 좋음)
- **별표(Ours)**: EywaAgent / EywaOrchestra / EywaMAS
- **원(Baseline)**: Single LLM Agent, Refine MAS, Debate MAS, MoA, X-MAS
- **녹색 화살표 "Better"**: 좌상단(고품질 + 저비용)이 더 좋은 방향
- **점선**: Pareto frontier

**해석:**
- **EywaAgent**: 가장 적은 토큰(~3000)으로 single LLM agent보다 훨씬 높은 utility. **거의 공짜 점심**.
- **EywaMAS / EywaOrchestra**: utility는 multi-agent baseline 중 최고이지만 토큰은 그들의 절반 이하.
- **Debate / MoA / X-MAS**: 토큰을 많이 쓰면서도 utility는 별로. 즉 **Pareto-dominated**.

→ **언어 기반 multi-agent의 토큰 비용은 대체로 utility 증가에 잘 안 비례하는데, Eywa는 이 trade-off 자체를 좌상단으로 끌어올림.**

## 4.4 견고성 — Hyperparameter 민감도

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/heterogeneous/image_6.png)

이 ablation들은 Eywa의 결과가 특정 세팅에 묶여있지 않다는 것을 보여줍니다.

**(a) LLM Sampling Temperature:** 0.0~1.0 변화에도 EywaAgent/MAS/Orchestra 모두 안정. 중간(0.4~0.6)에서 약간 peak.

**(b) FM (TabPFN) Softmax Temperature:** 0.1~0.9 변화에도 견고. FM 캘리브레이션 변화에도 strong.

**(c) Prompt Design:** Default → Detailed → Chain-of-Thought → ReAct 순으로 성능이 미세하게 상승. **더 구조화된 프롬프트일수록 약간씩 더 좋은 utility** — 즉 "도구를 언제 부를지" 추론이 명시적일수록 LLM이 FM을 잘 활용함.

**LLM backbone ablation (Table 2 요약):** gpt-4.1-nano → gpt-5-nano → gpt-5-mini로 갈수록 Utility 0.5680 → 0.6558 → 0.6640 우상향. **더 강한 LLM = 더 강한 Eywa**. Backbone 선택에 견고하면서도 더 큰 모델로부터 추가 이득을 받습니다.

# 5. 한 줄 정리

---

> **Eywa는 "LLM에게 도메인 FM을 호출할 수 있는 신경 다발(Tsaheylu)을 달아준다"는 단순한 아이디어를 MCP로 구현한 프레임워크다. 추가 학습 없이 토큰 30% 절감 + utility 6.6% 상승을 동시에 달성하며, conductor가 태스크별로 팀 구성을 자동 결정하는 EywaOrchestra가 가장 실용적인 형태다.**

**시사점:**
- **Agent 시스템에서 "언어가 universal interface"라는 가정 자체가 비효율의 원인일 수 있음.** 시계열, 표, 분자 같은 데이터를 자연어로 직렬화하는 것은 표현력 손실 + 토큰 폭증.
- **MCP 같은 표준 프로토콜**이 이 비대칭을 해소하는 핵심 인프라. 도메인 FM을 자연어 도구처럼 호출 가능하게 만들면, agent 생태계가 훨씬 풍성해짐.
- 향후 vision, geospatial, robotics 등 더 많은 modality-native FM이 같은 방식으로 합류하면, **"전문성 + 일반화 추론"의 진짜 협업**이 가능해질 것.

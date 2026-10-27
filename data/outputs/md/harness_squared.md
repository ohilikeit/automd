# 1. Introduction

---

### 1. 배경 및 문제 제기: Agent = Model + Harness

- LLM 에이전트의 성능은 모델만으로 결정되지 않습니다. system prompt, memory, skills, tools, subagent 같은 **harness**, 즉 모델을 감싼 실행 scaffolding에 따라 같은 모델도 결과가 크게 달라집니다.
- 에이전트가 코딩을 넘어 오피스·법률 업무까지 맡게 되면서 harness를 사람이 일일이 손보는 비용이 커졌습니다. 그래서 harness를 자동으로 개선하는 **Recursive Self-Improvement**(RSI)에 관심이 모입니다.
- 그런데 Meta-Harness 같은 초기 방법은 고정된 학습 과제와 unit test로 최적 harness를 미리 골라 배포합니다. 새로운 과제가 계속 들어오는 **open world**에서는 고정된 harness가 잘 통하지 않습니다.

Open world에서 harness를 진화시키기 어려운 이유는 둘입니다.

1. **정답 검증기가 없습니다.** 테스트 시점 과제는 처음 보는 것이라 정답 라벨도 믿을 만한 채점기도 없습니다. 기존 방법은 proxy signal, 가정한 정답, 출력 쌍 비교 등에 기대는데 이런 불완전한 증거로는 harness를 안전하게 고치기 어렵습니다.
2. **한 과제에서 얻은 개선이 다음 과제로 이어지지 않습니다.** 특정 과제에 맞춘 harness를 그대로 물려주면 분포가 바뀔 때 노이즈가 됩니다. 정작 재사용할 가치가 있는 건 **개선 과정 자체**입니다. 약점을 어떻게 진단하고 무엇을 시험했으며 결과가 어땠는지가 모두 여기 들어갑니다.

→ **핵심 질문: RSI로 "지금의 harness"와 "앞으로 harness를 고치는 방법"을 함께 개선할 수 있을까?**

### 2. 제안: Harness² (Recursive Agent Harnessing)

Harness²는 frozen 모델의 harness를 최적화하는 plug-and-play 프레임워크로, **두 단계의 재귀**로 움직입니다.

- **❶ Task-level recursion:** 지금 풀고 있는 과제의 **execution harness**를 고칩니다. 정답이 없으니 **Propose–Probe–Compose** 대조 단계를 씁니다. 수정안을 만들고 같은 조건에서 돌려 행동 차이를 본 뒤, 근거가 있는 수정만 합칩니다.
- **❷ Harnessing-level recursion:** 그 탐색 경험으로 **improvement harness**, 즉 harness를 고치는 쪽의 harness를 업데이트합니다. 이전 harness를 통째로 물려주지 않고 재사용할 만한 컴포넌트와 "언제, 왜 통했는지"라는 교훈만 라이브러리에 남깁니다.

> 이름의 "²"는 harness를 harnessing한다는 뜻입니다. 일하는 실력과 일 잘하는 법을 함께 키우려는 설계입니다.

### 3. 핵심 결과

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_1.png)

**왼쪽, WorkBuddy-Bench 도메인별 성능 (Gemini 3.5 Flash + OpenCode):** 회색은 base harness, 보라색은 가장 강한 baseline인 Meta-Harness, 파란 테두리는 Harness²($k=5$)입니다. Code 73.4 → **80.8**(+7.4), Web 69.9 → **78.3**(+8.4), Office 74.0 → **83.4**(+9.4)로 세 도메인 모두 Meta-Harness를 넘습니다. 별도 학습 과제로 미리 최적화한 harness를 테스트 과제 위에서 즉석으로 앞질렀습니다.

**오른쪽, WB-code에서 탐색 예산 $k$에 따른 변화:** Harness²는 $k$가 늘수록 꾸준히 올라 +4.3을 기록합니다. Harness Scaling과 ReasoningBank는 오르내림이 심합니다. 예산을 더 쓴다고 성능이 저절로 오르지는 않습니다.

- 개선 step 한 번($k=1$)만으로 WorkBuddy-Bench **최대 +11.5점**, JobBench **+4.5점**, LAB **+11.8%p**를 얻습니다.
- **executor와 editor가 같은 모델이어도** 모든 조합에서 성능이 오릅니다. 더 강한 외부 editor가 꼭 필요하지 않습니다.

# 2. Harness²: Harness 구성과 전체 파이프라인

---

## 2.1 Harness = 8개의 편집 가능한 파일

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_2.png)

모델 $\mathcal{M}$의 가중치는 고정하고 harness $\mathcal{H} = (c^1, \dots, c^8)$를 8개의 컴포넌트로 나눕니다. **각 컴포넌트가 하나의 파일**이라서 harness 수정은 곧 파일 diff가 됩니다.

| 그룹 | 컴포넌트 | 언제 모델에 닿나 | 역할 |
|---|---|---|---|
| **Context construction** | C1 System prompt | 매 호출 | 페르소나, 최상위 절차 |
| | C2 Guardrails | 매 호출 | 과제와 무관한 hard constraint |
| | C3 Memory | 매 호출 | 도메인 안에서 과제 간에 유지되는 사실 |
| | C6 Skills | 호출 시 | 필요할 때만 로드되는 전문 절차 |
| **Tool interface** | C4 Tool guidance | 매 호출 | 기존 도구 사용법 |
| | C7 Scripts | bash 실행 | 추론 대신 실행하는 결정론적 기능 |
| **Orchestration** | C5 Subagents | task tool | 별도 context에서 일하고 결과만 돌려줌 |
| **Middleware** | C8 Plugin | tool loop 내부 | tool call 전후 hook. 행동을 **지시하지 않고 강제** |

그림처럼 Context construction 파일이 모델 입력으로 들어가고 tool call은 C8 Plugin을 거쳐 Tool interface에 닿습니다. 결과도 plugin을 거쳐 돌아오며 작업을 위임할 때는 점선을 따라 C5 Subagent가 호출됩니다. 수단마다 강도도 다릅니다. 프롬프트(C1~C4)는 모델이 무시할 수 있지만 Script와 Plugin은 모델의 재량 밖에서 동작을 보장합니다. editor는 문제에 맞춰 이 중에서 고릅니다.

## 2.2 전체 파이프라인: 두 개의 재귀 루프

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_3.png)

과제 스트림 $q_1, \dots, q_N$을 정답 없이 순서대로 처리합니다. **모든 과제는 같은 초기 harness $\mathcal{H}_0$에서 출발합니다.**

**Task-Level Recursion (파란 영역, 과제 하나 안에서):**
1. **Agent Executor**가 초기 harness로 과제를 실행해 reference trajectory/output을 얻습니다.
2. **Harness Editor**가 이를 보고 **Invariant**(도메인 무관)와 **Domain**(도메인 특화) 후보 harness를 하나씩 만듭니다.
3. **Candidate Probes**에서 두 후보를 같은 과제에 각각 실행합니다.
4. **Harness Composer**가 결과를 비교해 근거 있는 수정만 합쳐 Final Harness를 만듭니다. 만들어진 harness는 다시 실행되어 최종 결과물을 냅니다.

**Harnessing-Level Recursion (보라 영역, 과제가 끝난 뒤):** Experience Reflector가 후보, probe, 조합 결정까지 포함한 개선 과정 전체를 돌아보며 교훈을 뽑아 **Improvement Harness**(라이브러리)에 반영합니다. 다음 과제는 다시 $\mathcal{H}_0$에서 시작하지만 갱신된 라이브러리의 도움을 받습니다.

> **execution harness는 개선 대상**이고 **improvement harness는 개선을 수행하는 장치**입니다. 후자는 과제 하나를 처리하는 동안 고정되어 있다가 과제가 끝날 때만 바뀝니다.

# 3. Task-Level Recursion: 정답 없이 harness 고치기

---

## 3.1 Propose → Probe → Compose

대조 단계 한 번을 식으로 쓰면 다음과 같습니다.

$$(\mathcal{H}', e', \xi) = \Phi(q_t, \mathcal{H}, \mathcal{E}; \mathcal{R}_t)$$

현재 harness $\mathcal{H}$와 실행 이력 $\mathcal{E}$를 받아 개선된 harness $\mathcal{H}'$, 그 실행 기록 $e'$, 단계 기록 $\xi$를 돌려줍니다. $\mathcal{R}_t$는 고정된 improvement harness입니다.

- **Propose:** 먼저 reference 실행을 얻습니다. reference로는 결과가 맞았는지 알 수 없지만 **harness가 어떻게 움직이는지**는 보입니다. 이를 바탕으로 범위가 다른 두 후보를 만듭니다.
  - **Invariant:** planning, tool 사용, context 관리처럼 도메인을 가리지 않고 반복되는 결함을 겨냥
  - **Domain-specific:** 현재 도메인의 지식, 절차, 제약을 겨냥
- **Probe:** 각 후보를 같은 과제, 같은 모델·환경에서 실행하고 reference와 비교해 **behavioral contrast**를 얻습니다. 수정이 결함을 줄였는지, 새로운 회귀를 만들었는지가 여기서 드러납니다. 정답 판정은 아니고 조합을 판단할 근거로만 씁니다. 실행마다 결과가 흔들리고 모델 편향도 공유하므로 이 증거를 과신하지 않도록 설계했습니다.
- **Compose:** 모든 증거를 읽고 지지되는 수정은 합치고 충돌하는 수정은 고치며 필요하면 현재 harness를 유지합니다. **예상 점수는 따지지 않고 구체적인 관찰만 판단 근거로 씁니다.**

> **채점표 없는 A/B 테스트**와 비슷합니다. 점수가 몇 점 올랐는지는 모르지만 "이 수정을 넣으니 에이전트가 입력 파일을 전부 확인하기 시작했다"는 변화는 눈에 보입니다. Harness²는 이런 행동 차이를 근거로 수정 여부를 정합니다.

## 3.2 Parallel vs. Sequential Recursion

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_4.png)

대조 단계를 예산 $k$만큼 확장하는 방식에는 parallel과 sequential이 있습니다.

- **(a) Parallel:** RSA의 aggregation 아이디어를 harness에 적용했습니다. reference 실행 $k$번, INV 후보 $k$개, DOM 후보 $k$개를 독립적으로 만들어 probe한 뒤, 모은 후보 풀 전체를 $k$개의 composition session이 각자 조합합니다. $k$가 **탐색의 폭**을 정합니다.
- **(b) Sequential:** Self-Refine처럼 대조 단계를 $k$번 이어 붙입니다. 화살표가 보여주듯 이전 단계의 harness와 누적 증거가 다음 단계로 넘어가므로 앞선 수정을 유지하거나 고치거나 되돌릴 수 있습니다. 여기서는 $k$가 **깊이**를 정합니다.

$k>1$이면 마지막에 agentic selector가 익명화된 결과물을 과제 요청과 대조해 하나를 고릅니다. 어떤 harness가 만들었는지는 보지 않고 새 내용을 합성하지도 않습니다. 고른 harness는 $\mathcal{H}_t^\star$로 표기합니다. 별표는 "검증된 최적"이 아니라 "**채택되었다**"는 뜻일 뿐입니다.

# 4. Harnessing-Level Recursion: 개선하는 방법을 개선하기

---

과제 $t$가 끝나면 그 개선 경험을 improvement harness에 반영합니다.

$$\mathcal{R}_{t+1} = \textsc{Update}(\mathcal{R}_t, \mathcal{H}_t^\star, \mathcal{D}_t)$$

- **Reflect:** 개선 기록 $\mathcal{D}_t$ 전체를 검토해 "어떤 수정이 어떤 조건에서 어떤 결함을 해결했나"를 추출합니다. **실행 대조로 뒷받침되는 교훈만** 남깁니다. 선택되었다는 사실만으로는 전이 가치를 인정하지 않습니다.
- **Update:** 라이브러리 엔트리 하나는 **실행 가능한 harness 트리 전체**와 `EXPERIENCE.md`로 이루어집니다. 컴포넌트를 글로 요약하지 않고 그대로 보존합니다.
- **Reuse:** 다음 과제는 여전히 $\mathcal{H}_0$에서 출발합니다. 이전 컴포넌트를 자동으로 물려받지 않고 Propose 단계에서 골라서 재사용하거나 변형합니다. Probe와 Compose 단계에서도 쌓인 경험을 참고합니다. 인덱스가 있어서 필요한 엔트리만 열어 볼 수 있습니다. 증거가 부족하면 새 엔트리를 아예 추가하지 않습니다.

> 매번 새 프로젝트를 빈 템플릿으로 시작하되 옆에 **"지난 프로젝트에서 통했던 해결책과 그 이유"를 정리한 사내 위키**를 펴 두는 것과 같습니다. 이전 설정을 통째로 복사하는 방식보다 분포 변화에 강합니다.

# 5. Experiments

---

## 5.1 실험 설정

- **벤치마크:** WorkBuddy-Bench(WB, 오피스형 산출물: Code/Web/Office), JobBench(실행이 긴 전문 직무), LAB(법률 업무, 기준 단위 통과율 Partial)
- **Executor × 초기 harness:** {Gemini 3.5 Flash, GLM-5} × {OpenCode, Codex}의 4개 구성. **Editor**는 Claude Opus 4.8이며 채점기나 rubric이 든 디렉토리에는 접근할 수 없습니다.
- **Baselines:** ReasoningBank(harness는 고정하고 과거 전략을 검색), Meta-Harness†(오프라인 탐색 후 배포), Harness Scaling(인스턴스마다 수정하고 상태는 넘기지 않음)
- 메인 표의 Harness²는 $k=1$이고 **+ Recursion**은 $k=5$ parallel에서 추가로 오른 양입니다. 라이브러리는 빈 상태에서 시작합니다.

## 5.2 메인 결과

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_5.png)

**WorkBuddy-Bench:** 위가 Gemini, 아래가 GLM-5이며 괄호 안은 3회 실행의 표준편차입니다.
- **네 구성 모두에서 Harness²가 Avg. 1위입니다.** 이득이 가장 큰 GLM-5+OpenCode는 **64.2 → 75.7**(+11.5)입니다. $k=5$로 늘리면 Avg.가 2.9~3.9 더 오릅니다.
- step 수도 대체로 줄었습니다. Gemini+OpenCode는 25.4 → 21.9입니다. 시행착오를 늘리지 않고 더 효율적으로 일해서 점수를 올렸습니다.
- baseline은 불안정합니다. ReasoningBank는 GLM-5+Codex Code에서 base보다 **6.6점 낮습니다**. 과거 전략을 검색해 context에 넣기만 하면 현재 과제와 맞지 않는 조언이 방해가 될 수 있습니다.

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_6.png)

**JobBench & LAB:** 왼쪽이 OpenCode, 오른쪽이 Codex입니다.
- JobBench에서는 Harness²가 네 구성 모두 1위입니다(Gemini+OpenCode 31.5 → **36.0**).
- LAB에서는 GLM-5+OpenCode가 59.6 → **71.4**(+11.8%p)이고 recursion으로 6.7%p가 더 붙습니다. 유일한 예외는 Gemini+Codex LAB으로, 여기서는 Meta-Harness가 근소하게 앞섭니다.
- Harness Scaling은 12개 조합 중 **7개에서 base보다 낮습니다.** 인스턴스마다 처음부터 고치면 노이즈에 쉽게 흔들립니다.

**약한 모델일수록 이득이 큽니다.** WB에서 개선된 GLM-5(75.7)는 base harness를 쓴 Gemini(72.4)를 넘어섭니다. **좋은 harness를 갖춘 약한 모델이 기본 harness를 쓰는 강한 모델보다 나을 수 있습니다.** 다만 JobBench에서는 반대로 Gemini의 이득이 더 컸습니다. 또 recursion이 초기 harness 간 격차를 늘 줄여 주지는 않습니다. JobBench에서는 OpenCode와 Codex의 차이가 0.5점까지 줄었지만, WB의 GLM-5에서는 0.2 → 1.3으로 벌어졌습니다.

## 5.3 Task-Level Recursion과 비용

![image_7](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_7.png)

- 회색 막대 **Best-of-$k$ base**는 base harness를 $k$번 돌려 가장 좋은 결과를 고른 값입니다. 연산만 더 쓰는 기준선이라고 보면 됩니다.
- **Parallel(파란 원)과 Sequential(보라 마름모) 모두 $k$에 따라 단조 증가합니다.** Parallel은 Code에서 80.8, Office에서 83.4까지 오릅니다.
- ReasoningBank와 Harness Scaling은 들쭉날쭉하고 Best-of-$k$조차 WB-code의 $k=5$에서 떨어집니다. 반면 **Harness²는 $k=2$만으로 모든 baseline의 $k=5$를 넘습니다.** $k=5$에서는 parallel이 앞서지만 Code의 $k=2$에서는 sequential이 더 높았습니다. 폭과 깊이 중 어느 쪽이 나은지는 예산 구간에 따라 다릅니다.

![image_8](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_8.png)

- 가로축은 과제당 총 연산량(M tokens, log scale, 개선 비용 포함), 세로축은 base 대비 향상폭입니다. 파란 계단선이 **Pareto frontier**입니다.
- **Harness²의 $k=1$과 $k=5$는 모두 frontier 위에 있습니다.** $k$를 1에서 5로 올리면 JobBench는 +4.5 → +7.5, WB는 +5.1 → +8.4가 됩니다. Meta-Harness(보라 사각형)는 Harness² $k=5$보다 연산을 더 쓰고도 성능이 낮습니다. 오프라인 탐색에 그만큼 비용이 많이 듭니다.

→ **탐색량을 늘리는 것만으로는 부족합니다. 탐색 결과를 근거에 따라 조합해야 성능이 꾸준히 오릅니다.**

## 5.4 Ablation

![image_9](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_9.png)

| 제거 항목 | 무엇이 빠지나 | WB | JobBench |
|---|---|---|---|
| **w/o library** | 이전 과제 경험을 쓰지 못함 | −3.1 | −1.8 |
| **w/o scoping** | invariant/domain 대신 범용 editor 2개 | −2.9 | **−3.7** |
| **w/o composition** | 조합하지 않고 후보 하나만 채택 | −1.6 | −1.9 |

세 요소 모두 기여합니다. **scoping** 결과를 보면 후보 수와 예산이 같아도 서로 다른 범위를 명시적으로 겨냥하게 해야 후보끼리 보완 관계가 생깁니다. composition을 빼면 두 제안 중 하나만 고르게 되므로 각 후보의 좋은 부분을 합치는 이득도 사라집니다.

## 5.5 라이브러리는 정말 쓸모가 있을까?

![image_10](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_10.png)

**왼쪽, Direct Editing w/o Execution:** reference 실행이나 probe 없이 base harness를 한 번에 직접 고치는 실험입니다.
- 라이브러리 없이 고치면(주황) **오히려 망가집니다.** Code는 73.4 → 63.2, Web은 69.9 → 65.0입니다. 실행 증거 없이 고치면 해가 됩니다. Propose–Probe–Compose 단계가 왜 필요한지도 여기서 거꾸로 드러납니다.
- $k=1$로 쌓은 라이브러리를 주면 base 수준으로 돌아오고, **$k=5$로 쌓은 라이브러리를 주면 실행 없이도 +0.9 / +4.4 / +5.1이 오릅니다.**

→ **task-level 탐색을 많이 할수록 라이브러리가 풍부해지고, 그 라이브러리는 probe 없이도 다음 개선을 돕습니다.** 두 재귀 단계는 실제로 맞물려 돌아갑니다.

**오른쪽, JobBench → WB 전이:** JobBench에서 쌓은 라이브러리를 WB에 쓰면 +0.6 / +0.5 / +2.9가 더 오릅니다. 각 과제 자신의 엔트리를 빼고 다시 돌리는 leave-one-out에서도 이득이 남으므로(WB +1.9), 과제를 외운 효과는 아닙니다.

![image_13](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_13.png)

- **왼쪽, 재사용 방식:** 라이브러리 코드를 50% 이상 그대로 가져온 경우(reused)는 0~10%에 그치고 변형해서 쓴 경우(adapted)가 16~43%입니다. 라이브러리는 복사해 쓰는 원본보다 **변형의 출발점**에 가깝습니다.
- **오른쪽, 수정이 몰리는 곳:** 모든 벤치마크에서 **C6 Skills**(28~38%)가 가장 많이 바뀌며 그다음이 System prompt와 Guardrails입니다. Web/Code에서는 Plugin, JobBench/Office에서는 Scripts 비중이 큽니다. 부록 ablation에서도 skills를 뺄 때 손실이 가장 컸습니다(WB −5.3). **필요할 때만 꺼내 쓰는 절차 지식이 harness 개선의 핵심 수단입니다.**

## 5.6 Executor = Editor

![image_11](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_11.png)

이번에는 executor 모델이 **자기 harness를 직접 고칩니다**(OpenCode, $k=2$). 회색이 base, 파란 테두리가 Harness²이며 위의 숫자가 향상폭입니다.
- **5개 모델 × 3개 벤치마크, 15개 조합 모두에서 향상됩니다.** GLM-5는 LAB에서 **+13.7**, 가장 강한 Claude Opus 5도 JobBench에서 **+5.5**를 얻습니다.
- 가장 강한 모델도 base 위로 더 올라갑니다. 모델이 강해져도 harness를 개선할 여지는 남습니다.

→ **더 강한 외부 editor 없이도, 모델은 자기 harness를 고쳐 스스로를 개선할 수 있습니다.**

## 5.7 Case Study: JobBench 과제 하나

![image_12](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/harness_squared/image_12.png)

**과제:** 일정표, 인력 명단, 예산 파일을 맞춰 서로 연결된 제작 문서를 만드는 일입니다. base는 **43.1점**이었고 실패 원인은 두 가지였습니다. 파일 간 수치를 대조하지 않았고 만든 산출물을 다시 열어 확인하지 않았습니다. **지식이 아닌 절차의 문제**였습니다.

**개선된 harness** (색칠된 C2, C4, C6, C7이 테스트 시점에 새로 쓴 부분):
- **Invariant 개입:** `guardrails.md`에 "모든 입력 파일을 먼저 목록화하고, 끝내기 전에 산출물을 다시 열어라. **파일이 존재한다고 내용이 맞는 건 아니다**"라는 규칙을 넣었습니다. 여기에 xlsx 수식 셀이 실제로는 빈칸으로 보이는지 잡아내는 `verify_outputs.py`를 함께 만들었습니다.
- **Domain 개입:** 일정·인력·예산 대조 절차를 담은 `SKILL.md`를 추가했습니다. 파일끼리 충돌할 때의 출처 우선순위, 확정 지출과 조건부 지출의 구분, 근무시간 상한 같은 규칙이 들어 있습니다. `tool_guidance.md`에는 "숫자가 `"$50"`, `"1,200"` 같은 문자열로 저장된 경우가 많으니 정리한 뒤 계산하라", "워크스페이스에 있는 사실은 웹에서 찾지 마라"는 내용을 넣었습니다.

**결과:** Composer가 네 수정을 모두 유지했고 점수는 **43.1 → 77.8**(+34.7)로 올랐습니다.

> "다시 확인하라"는 지시만 두지 않고 **확인을 실제로 해 주는 스크립트**까지 함께 만들었습니다. 지시와 실행 수단을 짝지어 넣었기 때문에 효과가 났습니다.

# 6. 정리

---

| 측면 | 기존 harness RSI | Harness² |
|---|---|---|
| 최적화 시점 | 오프라인, 또는 인스턴스별 단발 | 테스트 시점 과제 스트림에서 지속 |
| 피드백 | unit test, validation, proxy 정답 | **정답 없이** behavioral contrast |
| 과제 간 전이 | 이전 harness 상속, 또는 없음 | **개선 경험을 라이브러리에 쌓고 골라서 재사용** |
| 개선 대상 | execution harness | execution harness **+ improvement harness** |

- **한계:** behavioral contrast가 정답을 보장하지는 않습니다. 부록은 selector가 더 나쁜 후보를 고르거나 지나치게 제한적인 수정이 들어간 하락 사례 9건을 보고합니다. 또 parallel recursion으로 지연 시간에서 이득을 보려면 동시 실행 자원이 충분해야 합니다.

→ **Harness²는 에이전트가 일하면서 자기 작업 환경을 고치고 그 경험을 다음 일에 활용하는 루프를 정답 없이도 돌릴 수 있음을 보여줍니다. 모델 가중치를 건드리지 않고 harness 파일만 바꿔도 약한 모델이 강한 모델의 기본 성능을 넘어설 수 있다는 점에서 실무적으로도 의미가 큽니다.**

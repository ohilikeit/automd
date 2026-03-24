# 1. Introduction

---

### 1. 배경 및 문제 제기: Self-Improving AI의 근본적 한계

- **Self-improving AI**는 인간의 개입 없이 스스로 학습과 문제 해결 능력을 개선해 나가는 시스템을 의미합니다. 이러한 시스템이 실현되면 과학적 진보가 인간 속도에서 자율 가속 프로세스로 전환될 수 있습니다.
- **기존 접근법의 한계:** 대부분의 self-improving 시스템은 **고정된 meta agent**(기본 시스템을 수정하는 상위 시스템)에 의존합니다. 이는 meta agent의 설계가 곧 개선의 천장이 된다는 의미입니다. meta-meta 시스템을 추가해도 문제가 한 단계 위로 이동할 뿐, 무한 회귀에 빠집니다.
- **Darwin Gödel Machine (DGM)**은 코딩 도메인에서 open-ended self-improvement가 가능함을 보여주었습니다. 하지만 DGM의 **self-improvement 메커니즘(instruction-generation)은 고정되어 있고 수정 불가능**합니다. 이 메커니즘이 코딩 작업에 특화되어 있기 때문에, 코딩 외 도메인으로 일반화되지 못합니다.

→ **시를 잘 쓰게 된다고 해서, 자기 코드를 더 잘 고치게 되지는 않는다. DGM이 코딩에서 작동하는 것은, "평가 과제"와 "자기수정 과제"가 모두 코딩이라는 우연한 정렬(alignment) 덕분이다.**

### 2. 제안: HyperAgents — 자기참조적 자기개선 에이전트

핵심 아이디어는 **task agent(과제를 푸는 에이전트)와 meta agent(에이전트를 수정하는 에이전트)를 하나의 편집 가능한 프로그램으로 통합**하는 것입니다. 이 통합된 에이전트를 **hyperagent**라 부릅니다.

- **기존 DGM:** task agent와 meta agent가 분리됨 → meta agent가 고정 → 개선 메커니즘 자체는 개선 불가
- **HyperAgent:** task agent + meta agent가 하나의 프로그램 → **자기개선 메커니즘 자체도 수정 대상** → 이를 **metacognitive self-modification**이라 부름

> 비유하면, DGM은 "학습 방법이 고정된 학생"이고, HyperAgent는 "학습 방법 자체도 스스로 개선할 수 있는 학생"입니다. 시험 과목이 수학에서 역사로 바뀌어도, 후자는 새로운 과목에 맞는 학습 전략을 스스로 개발할 수 있습니다.

### 3. 핵심 결과

DGM에 hyperagent를 통합한 **DGM-Hyperagents (DGM-H)**는 다양한 도메인에서 task performance와 self-improvement 능력을 동시에 향상시킵니다:

- **코딩 (Polyglot):** 기존 DGM과 동등한 수준의 self-improvement 달성 (코딩 특화가 아님에도)
- **논문 리뷰:** test-set 정확도 0.0 → **0.710** (기존 open-source baseline 0.630 초과)
- **로보틱스 보상 설계:** 0.060 → **0.372** (기본 보상 함수 metric 0.348 초과)
- **올림피아드 수학 채점:** 도메인 간 transfer로 **imp@50 = 0.630** 달성
- **Meta-level 개선의 도메인 간 전이:** 논문 리뷰/로보틱스에서 학습한 자기개선 전략이 올림피아드 수학이라는 전혀 다른 도메인에서도 효과적
- **자기개선의 누적:** 한 run에서 학습한 개선이 다음 run에서 compound되어 축적

# 2. HyperAgents: 자기참조적 에이전트 프레임워크

---

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/hyperagent/image_1.png)

위 다이어그램은 기존 DGM과 DGM-H의 구조적 차이를 보여줍니다.

**상단 — Darwin Gödel Machine (DGM):**
1. **Archive:** 성공적인 coding agent 변이체들이 stepping stone으로 보관됩니다.
2. **Handcrafted instruction-generation:** 고정된, 사람이 설계한 메커니즘이 과거 성과와 코드를 분석하여 self-improvement instruction을 생성합니다.
3. **Self-modify → Evaluate on coding tasks:** 생성된 instruction에 따라 코드를 수정하고, 코딩 과제로 평가합니다.
4. 오른쪽의 **"aligned"** 표시는 핵심 제약을 보여줍니다: 평가 과제(코딩)와 self-modification 능력(코딩)이 동일 도메인이어야 개선이 작동합니다.

**하단 — DGM with HyperAgents (DGM-H):**
1. **Hyperagent = Task agent + Meta agent:** 하나의 편집 가능한 프로그램으로 통합되었습니다.
2. **Metacognitive Self-modify:** 고정 메커니즘 대신, hyperagent 자신의 코드(개선 절차 포함)를 자유롭게 수정합니다.
3. **Evaluate on computable tasks:** 코딩에 국한되지 않고 어떤 계산 가능한 과제든 평가 대상이 됩니다.
4. 오른쪽의 **"no need to be aligned"** 표시가 핵심: task 도메인과 self-modification 도메인의 정렬이 불필요합니다.

→ **meta agent가 hyperagent의 일부이므로, 자기개선 절차 자체가 개선 대상이 됩니다. 이것이 metacognitive self-modification의 본질입니다.**

## 2.1 핵심 개념 정의

- **Agent:** FM 호출, 외부 도구, 학습된 컴포넌트를 포함할 수 있는 **임의의 계산 가능한 프로그램**. Python으로 구현되며 Turing-complete합니다.
- **Task agent:** 주어진 과제를 해결하는 에이전트 (예: 코드 편집, 논문 리뷰 판정, 보상 함수 설계)
- **Meta agent:** 기존 에이전트를 수정하고 새 에이전트를 생성하는 에이전트
- **Hyperagent:** Task agent + Meta agent를 하나의 self-referential 프로그램으로 통합. 과제 해결 방법뿐 아니라 **미래 개선을 생성하는 방법 자체도 수정 가능**

## 2.2 Metacognitive Self-Modification

Hyperagent에서는 자기개선 메커니즘 자체가 수정 대상입니다:

- 과제 수행 개선과 **동시에** 향후 self-modification을 생성하고 적용하는 절차도 수정
- 기존 시스템(DGM 등)의 핵심 한계 — 고정된 meta-level 메커니즘 — 을 제거
- 결과적으로 hyperagent가 학습하는 것: (1) 과제를 더 잘 푸는 방법, (2) **자기개선을 더 잘 하는 방법**

## 2.3 DGM-H: Open-Ended Exploration + HyperAgents

DGM-H는 DGM의 open-ended exploration 구조를 유지하면서 hyperagent를 도입합니다:

1. **Archive:** 생성된 hyperagent 변이체들의 아카이브를 유지하며 시간에 따라 확장
2. **Metacognitive self-modification phase:** 선택된 부모 hyperagent들이 자신의 수정 버전을 생성
3. **Evaluation phase:** 수정된 hyperagent를 경험적으로 평가하여 아카이브에 추가
4. **Parent selection:** 성능에 비례하는 확률적 선택 + 자식 수에 반비례 (다양성 보존)

> DGM-H의 두 가지 핵심 구성요소: **(1)** DGM에서 물려받은 open-ended exploration, **(2)** 자기 생성 변이체를 통해 진화하는 초기 hyperagent. Meta-level 메커니즘을 수정 가능하게 함으로써 코딩을 넘어 일반화합니다.

# 3. Experiment Setup

---

## 3.1 초기 설정

- **초기 hyperagent:** frozen FM (GPT-4 계열) + bash tool + 파일 검사/수정 도구로 구성된 단순한 프로그램
- task agent는 단일 FM 호출로 과제를 풀고, meta agent는 에이전트 코드베이스의 어떤 부분이든 수정 가능
- 의도적으로 **단순한 초기 구현**을 유지: 넓은 탐색 공간을 확보하고, 가능한 개선 범위를 사전에 제약하지 않기 위함

## 3.2 Baselines

| Baseline | 설명 |
|---|---|
| **DGM-H** | 전체 시스템 (metacognition + open-ended exploration) |
| **DGM-H w/o self-improve** | Meta agent를 초기 상태로 고정 → 자기개선 메커니즘 비활성화 |
| **DGM-H w/o open-ended exploration** | Archive 없이 매번 이전 에이전트를 직접 교체 |
| **DGM** | 원본 DGM (코딩 특화 고정 instruction-generation) |
| **DGM-custom** | DGM의 instruction-generation을 각 도메인에 맞게 수동 커스터마이징 |

## 3.3 평가 도메인

4개의 다양한 도메인에서 평가합니다:

- **코딩 (Polyglot):** 코드 저장소 수정 벤치마크, 기존 DGM과 직접 비교
- **논문 리뷰:** AI 연구 논문의 accept/reject 이진 판정, 주관적 평가 도메인
- **로보틱스 보상 설계:** 자연어 설명 → 보상 함수 생성 → RL로 로봇 훈련 → 행동 품질 평가
- **올림피아드 수학 채점 (IMO-GradingBench):** 후보 풀이의 정확한 점수 예측, 도메인 간 transfer 평가용

# 4. Experiments

---

## 4.1 Task Performance 향상

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/hyperagent/image_2.png)

**왼쪽 — Training tasks 성능 곡선:**
- 가로축은 iteration 수(0~100), 세로축은 논문 리뷰 + 로보틱스 보상 설계의 combined 성능입니다.
- **파란 실선 (DGM-H):** 꾸준히 우상향하며 iteration 100에서 약 0.75에 도달합니다. 음영 영역은 5회 반복의 95% 신뢰구간입니다.
- **보라색 (DGM-custom):** 도메인별 수동 커스터마이징으로 빠르게 성장하나, 후반에 DGM-H에 추월당합니다.
- **초록색 (DGM-H w/o open-ended exploration):** Archive 없이는 거의 개선이 이루어지지 않습니다.
- **분홍색 (DGM-H w/o self-improve):** Self-improvement 없이도 일부 진전이 있지만 DGM-H보다 현저히 낮습니다.
- **회색 (DGM):** 코딩 외 도메인에서는 거의 개선되지 않습니다.

**오른쪽 상단 — Paper Review test-set 성능:**
- DGM-H가 **0.710**으로 가장 높고, representative baseline (AI-Scientist-v2)의 0.630을 초과합니다.
- 원본 DGM은 0.0으로 논문 리뷰에 전혀 적용되지 못합니다.
- DGM-custom은 0.590으로 수동 커스터마이징 덕분에 양호하나 DGM-H에 미치지 못합니다.

**오른쪽 하단 — Robotics Reward Design test-set 성능:**
- DGM-H가 **0.372**로 default reward function (standing tall, 0.348)을 초과합니다.
- DGM-custom은 0.348로 local optimum(서 있기)에 갇히는 경향을 보입니다. 반면 DGM-H는 **jumping behavior를 유도하는 보상 함수**를 발견하여 local optimum을 탈출합니다. "로봇 몸통 높이를 최대화하라"는 test task에서, standing tall은 쉬운 해지만 점프가 더 높은 점수를 받습니다. DGM-H만이 이 비자명한(non-trivial) 전략을 발견합니다.

**핵심 발견:**

- **Metacognitive self-modification과 open-ended exploration 모두 필수적:** 둘 중 하나라도 제거하면 성능이 크게 하락합니다 (p < 0.05).
- **DGM-H는 코딩 특화가 아님에도** 코딩 도메인에서 원본 DGM과 동등한 self-improvement를 달성합니다.
- DGM-H는 표면적 프롬프트 수정을 넘어 **구조화된 의사결정 파이프라인**(체크리스트, 판정 규칙, 명확한 기준)을 자율적으로 구축합니다.

## 4.2 자기개선 능력(Ability to Improve)의 향상

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/hyperagent/image_3.png)

이 실험은 **meta agent의 능력이 시간에 따라 향상되는지**, 그리고 그 향상이 **도메인 간 전이(transfer)** 되는지를 검증합니다. 핵심 metric은 **imp@50** (improvement@50): 초기 task agent 대비 50회 iteration 내 최고 성능 향상폭입니다.

**왼쪽 — Olympiad-level math grading training curve:**
- 가로축은 iteration(0~50), 세로축은 IMO-level grading 성능입니다.
- 초기 에이전트(빨간색, 녹색)는 거의 개선되지 않습니다.
- **DGM-H w/o self-improve + transfer (녹색 점선):** 논문 리뷰/로보틱스에서 온 transfer hyperagent를 시작점으로 사용하면 빠르게 0.5~0.6까지 성장합니다.

**중앙 — imp@50 비교 (DGM 계열):**
- Initial agent, DGM w/o self-improve, transfer agents from prior DGM, DGM w/o self-improve + transfer 모두 **imp@50 = 0.0**
- → DGM의 고정된 instruction-generation은 도메인 간 전이 가능한 meta-level 개선을 학습하지 **못합니다**.

**오른쪽 — imp@50 비교 (DGM-H 계열):**
- Initial agent와 DGM-H w/o self-improve도 **imp@50 = 0.0** (초기 meta agent는 수학 채점에서 개선 능력 없음)
- **Transfer agents from prior DGM-H + DGM-H w/o self-improve:** **imp@50 = 0.630**
- → 논문 리뷰/로보틱스에서 진화한 hyperagent가 올림피아드 수학이라는 전혀 다른 도메인에서도 효과적으로 개선된 task agent를 생성합니다.

→ **DGM-H는 "특정 도메인을 잘 푸는 법"이 아니라, "어떤 도메인이든 자기개선하는 법"을 학습합니다. 이것이 metacognitive self-modification의 핵심 가치입니다.**

Transfer hyperagent가 학습한 범용 meta-level 능력의 예:
- **Performance tracking:** 세대별 성능을 기록하고 추세를 분석하는 인프라를 자율적으로 구축합니다. 아래는 DGM-H가 스스로 생성한 `PerformanceTracker` 클래스의 예시입니다:

```python
class PerformanceTracker:
    """Tracks performance metrics across agent generations."""
    def record_generation(self, generation_id, domain, score, metadata=None):
        """Record performance for a generation."""
        entry = {"generation_id": generation_id, "domain": domain,
                 "score": score, "timestamp": datetime.now().isoformat()}
        self.history.append(entry)

    def get_improvement_trend(self, domain=None, window=5):
        """Calculate improvement trend using moving average."""
        recent_avg = sum(h['score'] for h in filtered[-window:]) / window
        older_avg = sum(h['score'] for h in filtered[-window*2:-window]) / window
        return recent_avg - older_avg  # Positive if improving
```

- **Persistent memory:** 단순 수치 로깅을 넘어 합성된 인사이트, 인과 가설, 미래 계획을 JSON 형태로 저장하고 이후 세대에서 적극 참조합니다. 예를 들어, "Gen55는 정확도가 높지만 reject recall이 너무 가혹하다 → Gen64의 균형 잡힌 추론과 결합해야 한다"와 같은 전략적 분석이 메모리에 축적됩니다.

→ **이러한 메타 능력은 특정 도메인에 종속되지 않습니다. "성능을 추적하고, 실패를 분석하고, 전략을 조정하는" 능력은 논문 리뷰든 수학 채점이든 동일하게 적용됩니다.**

## 4.3 자기개선의 누적 (Compounding Self-Improvements)

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/hyperagent/image_4.png)

이 실험은 한 setting에서 학습한 self-improvement가 다른 setting에서 **계속 누적(compound)**될 수 있는지를 검증합니다.

**왼쪽 — Olympiad-level math grading training curve (200 iterations):**
- **파란색 (DGM-H):** 초기 에이전트에서 시작, 200 iteration 후 약 0.61 도달
- **주황색 (DGM-H + transfer):** 이전 DGM-H run의 transfer hyperagent에서 시작, **더 빠른 초기 성장**과 더 높은 최종 성능(약 0.64) 도달
- **녹색 점 (ProofAutoGrader):** 기존 representative baseline (0.670)
- **초록색 (DGM-H + transfer + from ProofAutoGrader):** ProofAutoGrader를 task agent로 사용하고 transfer meta agent를 결합, **0.700**으로 baseline을 초과

**오른쪽 — Test-set 성능 비교:**
- DGM-H: **0.610**
- DGM-H + transfer: **0.640**
- DGM-H + transfer + from ProofAutoGrader: **0.700** (representative baseline 0.670 초과)

→ **이전에 학습한 자기개선 전략이 새로운 도메인에서 재사용되고 누적됩니다.** 높은 성능 수준에서의 추가 개선(0.7→0.8)은 낮은 수준(0.0→0.1)보다 본질적으로 어렵다는 점에서, 이 결과의 의미가 더 큽니다.

> 전체 IMO-GradingBench에서 평가하면, DGM-H는 ProofAutoGrader의 정확도를 0.561 → 0.601로, mean absolute error를 0.178 → 0.175로 개선합니다.

# 5. Safety Discussion

---

DGM-H는 자율적으로 자신의 행동과 개선 메커니즘을 수정할 수 있기 때문에 고유한 안전성 고려가 필요합니다.

**실험에서의 안전 조치:**
- 모든 에이전트 생성 코드는 **sandboxed 환경**에서 실행 (리소스 제한, 타임아웃, 인터넷 접근 제한)
- 사전 정의된 과제와 metric으로만 평가 수행
- 전 실험에 걸쳐 **human oversight** 유지

**핵심 안전 이슈 — 인간 감독 속도를 초과하는 진화:**
- AI 시스템이 점점 더 open-ended하게 자기수정하게 되면, 인간이 감사(audit)하거나 해석할 수 있는 속도보다 빠르게 진화할 가능성
- 현재 DGM-H는 안전한 연구 경계(sandbox, 통제된 평가) 내에서 작동하지만, 이러한 안전장치가 시스템 능력 증가에 따라 불충분해질 수 있음
- **투명성, 감독 가능성, 사회적 합의**가 이러한 시스템의 책임 있는 배포를 위한 핵심 과제

**현재 한계:**
1. 고정된 task distribution에서 작동 (과제 자체를 생성하지는 않음)
2. Open-ended exploration의 일부 구성요소(parent selection, 평가 프로토콜)는 여전히 고정
3. 이러한 outer-loop 구성요소까지 hyperagent가 수정할 수 있게 하는 것이 향후 연구 방향

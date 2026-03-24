# 1. Introduction

---

**"Long-running Claude for Scientific Computing"**은 Anthropic Discovery 팀의 Siddharth Mishra-Sharma가 작성한 글로, Claude Code를 활용하여 **수일간의 자율 AI 워크플로우**를 과학 컴퓨팅에 적용하는 실전 가이드를 제공합니다.

### 1. 배경 및 문제 제기: AI 에이전트의 과학 연구 활용

- **기존 패러다임의 한계:** 대부분의 과학자들은 AI 에이전트를 **대화형(conversational)**으로 사용합니다. 각 단계를 직접 관리하며 한 번에 하나의 작업을 지시하는 방식입니다. 이는 마치 실험실에서 조교에게 한 단계씩 구두로 지시하는 것과 같습니다.
- **새로운 접근:** 모델이 **장기 과제(long-horizon tasks)**에서 뛰어난 성능을 보이면서, 고수준 목표만 지정하고 에이전트가 자율적으로 작업을 수행하는 패러다임이 등장했습니다. 이를 통해 **수주~수개월 걸리던 프로젝트를 수시간~수일 만에** 완료할 수 있습니다.
- **적합한 과학 작업:** 수치 솔버(numerical solver) 구현, 레거시 소프트웨어의 현대 언어 변환, 대규모 코드베이스 디버깅 등 **명확한 성공 기준이 있는 체계적 작업**이 이 모델에 적합합니다.
- **선행 사례:** Anthropic의 **C 컴파일러 프로젝트**는 2,000개 세션에서 병렬적으로 에이전트를 활용하여 Linux 커널을 컴파일할 수 있는 C 컴파일러를 구축했습니다.

### 2. 실험 대상: 미분 가능한 우주론적 Boltzmann Solver

이 글에서 다루는 구체적 프로젝트는 **JAX로 구현하는 미분 가능한 우주론적 Boltzmann solver**입니다.

- **Boltzmann solver란?** 초기 우주에서 광자(photon), 바리온(baryon), 뉴트리노(neutrino), 암흑물질(dark matter)의 결합 방정식을 시간에 따라 풀어, **우주 마이크로파 배경복사(CMB)**의 통계적 성질을 예측하는 수치 코드입니다. CLASS와 CAMB가 이 분야의 표준 인프라로, Planck과 Simons Observatory 등 서베이 데이터를 이용한 우주론 모델 제약에 사용됩니다.
- **왜 미분 가능해야 하는가?** 기존 Boltzmann solver는 미분이 불가능하여 파라미터 추정 시 MCMC 등 비효율적인 방법에 의존합니다. **미분 가능(differentiable)** 버전을 만들면 **gradient 기반 추론**으로 파라미터 추정 속도를 극적으로 높일 수 있습니다. JAX는 자동 미분과 가속기 호환성을 자연스럽게 제공합니다.
- **난이도:** 이 작업은 저자의 핵심 전문 분야 밖에 있었습니다. 해당 전문성을 갖춘 연구 그룹들이 CLASS의 일부 기능만 JAX로 구현하는 데에도 **수개월~수년의 연구자 시간**이 소요되었습니다.
- **C 컴파일러와의 구조적 차이:** C 컴파일러가 병렬화 가능한 반면, Boltzmann solver는 **깊은 결합(deeply coupled)** 구조를 가집니다. 초기 우주 recombination 단계의 작은 수치 오차가 이후 모든 결과에 미묘하게 영향을 미치므로, **순차적 작업, 전체 인과 체인을 통한 디버깅, 도메인 지식 적용**이 필요합니다.

→ **"비전문가가 AI 에이전트의 최소한의 조향(steering)만으로 전문 분야의 복잡한 수치 코드를 구현할 수 있는가?"라는 질문에 대한 실험입니다.**

# 2. 에이전트 운용 전략

---

## 2.1 CLAUDE.md: 계획 수립과 반복적 개선

자율 에이전트 연구팀을 관리하는 핵심은 **명확한 지시서(instruction)를 작성하는 데 대부분의 시간을 투자**하는 것입니다.

- 프로젝트 루트에 **`CLAUDE.md`** 파일을 배치합니다. Claude는 이 파일을 특별하게 취급하며, 작업 중 지속적으로 참조합니다.
- 이 파일에는 다음이 포함됩니다:
  - 프로젝트 전체 목표와 설계 결정사항
  - 정확도 목표 (예: **CLASS 대비 0.1% 이내** — 이는 CLASS와 CAMB 간 전형적 오차 수준)
  - 기술적 맥락 정보
- **핵심:** Claude가 작업을 진행하면서 이 파일을 **직접 수정**할 수 있습니다. 즉, 계획은 고정된 것이 아니라 에이전트와 함께 진화합니다.

## 2.2 CHANGELOG.md: 세션 간 기억 유지

에이전트의 **이동식 장기 기억(portable long-term memory)** 역할을 하는 진행 파일입니다. 마치 실험실 노트(lab notes)와 같습니다.

좋은 진행 파일이 추적해야 하는 것:
- **현재 상태**와 완료된 작업
- **실패한 접근법과 그 이유** — 이것이 매우 중요합니다. 없으면 다음 세션에서 동일한 실패를 반복합니다.
  - 예시: *"Tsit5로 perturbation ODE를 풀려고 시도했으나 시스템이 너무 stiff함. Kvaerno5로 전환."*
- 주요 체크포인트에서의 **정확도 테이블**
- 알려진 **한계점**

> **실패한 접근법의 기록이 없으면, 이후 세션에서 동일한 막다른 길을 반복 시도합니다.** 이는 인간 연구자의 실험 노트에서 "이건 안 됐다"라고 기록하는 것과 같은 원리입니다.

## 2.3 Test Oracle: 객관적 진행 지표

장기 자율 과학 연구에서 에이전트는 **자신이 진전하고 있는지 객관적으로 판단**할 수 있어야 합니다.

선택지:
- **레퍼런스 구현** (이 프로젝트에서는 CLASS C 소스 코드)
- **명확히 정량화 가능한 목표**
- **기존 테스트 스위트**

에이전트에게 테스트 스위트를 **확장하고 지속적으로 실행**하도록 지시합니다. 이를 통해 regression을 방지합니다.

→ **"CLASS C 소스를 레퍼런스 구현으로 사용하여 unit test를 구성하고 지속적으로 실행하라"는 지시가 CLAUDE.md에 포함됩니다.**

## 2.4 Git을 통한 조율

Git은 에이전트의 진행 상황을 **비동기적으로 모니터링**하는 핵심 도구입니다.

실질적으로 `CLAUDE.md`에 포함되는 지시:
- *"의미 있는 작업 단위마다 commit & push"*
- *"매 커밋 전에 `pytest tests/ -x -q` 실행"*
- *"기존에 통과하던 테스트를 깨뜨리는 코드는 절대 커밋하지 않기"*

이를 통해:
- 복구 가능한 히스토리 확보
- 로컬에서 진행 상황 확인 가능
- 컴퓨트 할당 시간 만료 시 작업 손실 방지

## 2.5 실행 루프: HPC 클러스터에서의 운용

실행 환경은 **SLURM 스케줄러를 사용하는 HPC 클러스터**이지만, 핵심 아이디어(진행 파일, 테스트 오라클, 명확한 프롬프트)는 **어떤 환경에서든** 적용됩니다.

**실행 흐름:**
1. 로컬에서 계획을 반복 수정하여 `CLAUDE.md`에 인코딩
2. 컴퓨트 노드에서 **tmux 세션** 안에 Claude Code를 실행
3. 세션을 분리(detach)하고 비동기적으로 진행 상황 확인
4. 필요 시 재접속하여 방향 조정 또는 새 작업 부여

```bash
#!/bin/bash
#SBATCH --job-name=claude-agent
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-32:1
#SBATCH --time=48:00:00
#SBATCH --output=agent_%j.log
cd $PROJECT/my-solver
source .venv/bin/activate
export TERM=xterm-256color
tmux new-session -d -s claude "claude; exec bash"
tmux wait-for claude
```

> 세션을 분리해 두면 **노트북을 닫고 커피를 마시면서 핸드폰으로 GitHub 커밋을 확인**하는 식의 비동기적 연구가 가능합니다.

## 2.6 The Ralph Loop: 에이전트 나태함 극복

모델이 아무리 뛰어나도 복잡한 멀티파트 작업에서 **"에이전트 나태함(agentic laziness)"**을 보일 수 있습니다. 완료되지 않은 작업에서 핑계를 대며 멈추는 현상입니다.

> *"오늘은 늦었으니 내일 다시 하면 안 될까요?"*

**Ralph loop**는 이를 극복하는 오케스트레이션 패턴입니다:
- 에이전트가 완료를 선언하면 **다시 컨텍스트에 투입**
- *"정말 끝났는가?"* 확인
- 사양을 충족할 때까지 **반복 수행**

```
/ralph-loop:ralph-loop "Please keep working on the task until the success criterion
of 0.1% accuracy across the entire parameter range is achieved."
--max-iterations 20 --completion-promise "DONE"
```

Claude는 **최대 20회 반복**하며, 작업이 완전히 완료되어 "DONE"을 선언할 때까지 계속 작업합니다.

# 3. Results

---

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/long_running_claude_for_scientific_computing/image_1.png)

위 그래프는 Claude가 처음부터(from scratch) 수일에 걸쳐 작업하며 **CLASS 레퍼런스 구현 대비 정확도를 개선해 나간 궤적**을 보여줍니다.

**축 해석:**
- **Y축** (로그 스케일): $|C_\ell^{\text{JAX}} / C_\ell^{\text{CLASS}} - 1|$ — JAX 구현과 CLASS 간 상대 오차. 낮을수록 좋으며, **1% 이하(0.01 이하)**가 목표입니다.
- **X축**: 날짜 (2월 7일 금요일 ~ 2월 12일 수요일, 약 5일)
- **세 개의 선**: CMB angular power spectra의 세 가지 주요 출력
  - **빨간색** $C_\ell^{TT}$ (온도-온도): CMB의 가장 기본적인 파워 스펙트럼
  - **파란색** $C_\ell^{EE}$ (편광 E모드-E모드)
  - **초록색** $C_\ell^{TE}$ (온도-편광 교차)
- **보라색 ×**: dead end(막다른 길)로 revert한 지점

**개발 궤적 분석:**

1. **2월 7일 (금)**: Source function normalization fix로 시작. 초기 오차는 **100~1000%** 수준으로 극도로 부정확합니다.
2. **2월 7일 저녁**: Sign flip 시도 → **revert** (실패한 접근). 이후 Bessel function + k-grid resolution 개선으로 $C_\ell^{TT}$가 약 10% 수준으로 하락.
3. **2월 8일 (토)**: "17% floor" 발견 — source function bug가 원인. 이후 baryon velocity + tight-coupling fix로 **1% 근처**까지 도달.
4. **2월 9일 (일)**: Source interpolation + recombination rewrite, 그리고 radiation damping 시도(효과 없음으로 **revert**). 이 시점에서 $C_\ell^{TT}$는 **0.1~1%** 범위.
5. **2월 10~11일**: 점진적 개선. Radiation streaming in Einstein equations 수정.
6. **2월 11~12일**: Full neutrino hierarchy 구현, neutrino energy-momentum fix로 최종적으로 **세 가지 스펙트럼 모두 sub-percent(< 1%) 정확도** 달성.

**핵심 관찰:**
- 개발 과정은 **다소 비효율적(clunky)**이었습니다. 한동안 **단일 fiducial parameter point에서만 테스트**하여 버그 탐지 범위가 좁았습니다.
- gauge convention 혼동이나, 우주론 전문가라면 즉시 발견할 수 있는 실수에 **수시간을 소비**하기도 했습니다.
- 그럼에도 **sub-percent 정확도를 향한 꾸준한 진전**이 이루어졌습니다.

> **예상치 못한 부수효과:** 저자는 자신의 핵심 전문 분야가 아님에도, Git 커밋 히스토리를 따라가며 Boltzmann solver와 그 물리학에 대해 **상당한 지식을 습득**했습니다. Claude의 점진적 진행 과정을 따라가며 모르는 개념을 찾아보는 것이 효과적인 **"과학 삼투(science osmosis)"**가 되었습니다. 커밋 로그는 마치 **빠르고 극도로 문자 그대로인 포스닥의 실험 노트**와 같았습니다.

## 한계와 시사점

결과물이 아직 **production-grade는 아닙니다.** 모든 파라미터 regime에서 CLASS와 허용 가능한 정확도 매칭이 되지는 않습니다. 그러나 이 실험이 보여주는 것은 분명합니다:

- **압축:** 에이전트 기반 개발이 **연구자 수개월~수년의 작업을 수일로 압축**할 수 있음을 시연
- **접근성:** 비전문가가 최소한의 도메인 지식으로도 전문 분야의 수치 코드를 상당 수준까지 구현 가능
- **부수 학습:** 에이전트의 커밋 히스토리를 따라가는 것 자체가 효과적인 학습 경험

> AI 연구에서 실험을 밤새 돌려놓고 아침에 결과를 확인하는 것이 보편적입니다. 저자는 이제 **에이전트를 돌려놓지 않는 밤**도 같은 기회비용을 가진다고 주장합니다.

→ **"컴퓨트와 명확한 성공 기준이 있는 프로젝트에서, 에이전트를 돌리지 않는 매 밤은 테이블 위에 놓아둔 잠재적 진전(potential progress left on the table)이다."**

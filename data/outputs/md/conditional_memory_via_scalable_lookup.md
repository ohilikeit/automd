# 1. Introduction

---

### 1. 배경 및 문제 제기: '연산'으로 '기억'을 흉내 내는 비효율성

- **MoE의 한계:** 현재 LLM은 **조건부 연산(Conditional Computation)**을 활용하는 MoE(Mixture-of-Experts)가 표준으로 자리 잡았습니다. 하지만 언어 모델링은 '복합적 추론(Compositional Reasoning)'과 '지식 검색(Knowledge Retrieval)'이라는 성격이 다른 두 가지 작업으로 나뉩니다
- **구조적 비효율:** 고유 명사나 관용구 같은 정적이고 국소적인 패턴(Static patterns)을 처리하기 위해 Transformer의 깊은 레이어와 연산을 소모하는 것은 비효율적입니다. 이는 마치 정적인 조회 테이블(Lookup table)을 런타임에 값비싼 연산으로 재구성하는 것과 같으며, 고차원적 추론에 쓰여야 할 리소스를 낭비합니다.
    
    → **트럼프 라는 단어 뒤에 대통령이 올 확률을 계산하기 위해 수많은 뉴런을 거치는 것은 낭비다.** 
    

### 2. 제안: Engram (조건부 메모리)

- **새로운 희소성 축(Axis of Sparsity):** 동적인 로직을 처리하는 '조건부 연산(MoE)'과 상호보완적인 개념으로, 정적인 지식을 O(1)조회로 처리하는 **'조건부 메모리(Conditional Memory)'**를 제안합니다.
    
    **→ 국소 패턴이나 고정된 지식을 정적인 임베딩 테이블에 저장해두고 필요할 때 즉시 찾아쓰기(lookup)**
    
- **Engram 모듈:** 고전적인 N-gram 방식을 현대적으로 재해석하여 토크나이저 압축, 멀티 헤드 해싱, 문맥 인식 게이팅(Context-aware gating) 등의 기술을 적용한 **Engram**을 설계했습니다.

### 3. 실험 결과: 추론 능력 및 효율성 향상

- **희소성 할당(Sparsity Allocation)의 발견:** 파라미터 예산이 고정되었을 때, 모든 용량을 MoE에 할당하는 것보다 Engram 메모리와 적절히 배분할 때 성능이 최적화되는 **U자형 스케일링 법칙**을 발견했습니다.
- **성능 우위:** 동일한 파라미터 및 연산량(Iso-FLOPs)을 가진 MoE 베이스라인 대비, Engram-27B 모델이 더 뛰어난 효율성을 보였습니다.
- **추론 능력 강화:** 단순 지식 관련 태스크(MMLU +3.4)뿐만 아니라, **일반 추론(BBH +5.0), 코딩 및 수학(HumanEval +3.0, MATH +2.4)** 영역에서 더 큰 성능 향상을 확인했습니다.

### 4. 메커니즘 분석 및 시스템 효율성

- **유효 깊이(Effective Depth) 증가:** Engram이 초반 레이어에서 정적인 지식 재구성을 전담함으로써, 모델의 **깊은 레이어들이 복잡한 추론에 집중할 수 있게 합니다.**
- **Long Context 성능:** 국소적인 의존성 처리를 Engram에 위임함으로써 Attention 용량을 글로벌 문맥 파악에 집중시켜 긴 문맥 처리 성능(NIAH 등)을 대폭 향상시켰습니다.
- **인프라 효율성:** 동적 라우팅을 하는 MoE와 달리, Engram은 결정론적 ID를 사용하여 런타임 프리패칭(Prefetching)이 가능합니다. 100B 규모의 테이블을 호스트 메모리(CPU RAM)로 오프로딩해도 오버헤드가 3% 미만에 불과하여 GPU 메모리 제약을 효과적으로 극복했습니다.

# 2. 구조

---

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_1.png)

## 2.1 Sparse Retrieval via Hashed N-grams

 **Engram이 어떻게 방대한 N-gram 패턴을 메모리 폭발 없이 저장하고, O(1)의 속도로 찾아내는가?**

### 2.1.1 토크나이저 압축 (Tokenizer Compression): 키(Key)의 정규화

가장 먼저 수행하는 작업은 검색의 'Key'가 될 텍스트 패턴을 최적화하는 것입니다.

- 문제점 (Raw Token ID의 비효율성):기존 LLM의 토크나이저(BPE 등)는 **원문 복원(Lossless reconstruction)**이 주목적입니다.
    - 그래서 의미는 같지만 표기가 다른 단어들(예: Apple vs apple)에 서로 다른 ID를 부여합니다.
    - 이는 검색 관점에서 보면 '캐시 적중률(Hit rate)'을 떨어뜨리고 불필요한 엔트리를 늘리는 원인이 됩니다.
- 해결책 (Vocabulary Projection): Engram은 **어휘 투영 레이어(Vocabulary Projection Layer)**를 도입합니다.
    - 단순히 토큰 ID를 쓰는 게 아니라, **정규화된 형태(Canonical form)**로 매핑하는 전사 함수를 거칩니다.
    - 예를 들어, 대소문자 통일, 유니코드 정규화(NFKC) 등을 수행하여 `Apple`과 `apple`을 동일한 'Canonical ID'로 변환합니다.
- 효과: 이 과정을 통해 128k 크기의 토크나이저 기준으로 실질적인 어휘 크기(Effective Vocabulary Size)를 약 23% 줄였습니다. 이는 메모리 슬롯의 '의미적 밀도(Semantic Density)'를 높이는 효과를 줍니다.

### 2.1.2 멀티 헤드 해싱 (Multi-Head Hashing): 충돌 회피 및 공간 효율화

압축된 토큰들이 N-gram(예: "Alexander the Great")을 형성하면, 이 조합의 수는 기하급수적으로 늘어납니다. 이를 물리적인 테이블에 전부 1:1로 매핑하는 것은 불가능합니다. Engram은 이를 **해싱(Hashing)**으로 해결합니다.

- 해싱 기반 조회 (Hashing-based Lookup): Engram은 N-gram 텍스트 자체를 Key로 하여 임베딩 테이블(Value)을 조회합니다. 이때 별도의 인덱싱 자료구조(Tree 등)를 타지 않고, **결정론적 해시 함수(Deterministic Hash Function)**를 통해 곧바로 메모리 주소를 계산합니다.
- 멀티 헤드 전략 (Multi-Head Strategy): 단일 해시 함수를 쓰면 서로 다른 N-gram이 같은 슬롯을 가리키는 **해시 충돌(Collision)**이 필연적으로 발생합니다. 이를 완화하기 위해 Engram은 **K개의 서로 다른 해시 헤드(Hash Heads)**를 사용합니다.
    - **작동 원리:** 하나의 N-gram에 대해 K개의 서로 다른 해시 함수를 돌려, K개의 임베딩 벡터를 가져옵니다.
    - **벡터 결합:** 이렇게 가져온 K개의 벡터들을 **연결(Concatenation)**하여 최종 메모리 벡터 e_t를 구성합니다.
    - **의의:** 마치 블룸 필터(Bloom Filter)나 멀티 헤드 어텐션(Multi-head Attention)의 개념과 유사하게, **정보를 여러 해시 공간에 분산 저장**함으로써 특정 해시 충돌이 발생하더라도 다른 헤드의 정보가 이를 보정하여 노이즈를 상쇄할 수 있게 설계되었습니다.

## 2.2 Context-aware Gating

**가져온 정보가 현재 문맥에 진짜 맞는 정보인지 검증하고 필터링하는 단계.**

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_2.png)

### **2.2.1 메커니즘: Hidden State가 '검사관'이 된다**

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_3.png)

Engram은 트랜스포머의 Self-Attention 구조(Query-Key-Value)를 차용하여 이 문제를 우아하게 해결합니다.

- **Query (h_t):** 현재 모델의 **Hidden State**입니다. 이전 레이어들을 거치며 글로벌 문맥을 이미 학습한 상태로, "내가 지금 필요한 문맥"을 알고 있는 주체입니다.
- **Key & Value (e_t):** 해싱으로 찾아온 **N-gram 임베딩**입니다. 정적인 정보 저장소입니다.
- **Gating Score 계산 (alpha_t):**
    - 현재 문맥(Query)과 찾아온 정보(Key)의 내적(Dot Product)을 구한 뒤 시그모이드(Sigmoid)를 씌웁니다.
    - **의미:** "지금 내 문맥(h_t)이랑 이 검색 결과(e_t)랑 얼마나 관련 있어?"를 수치화한 것입니다.

### **2.2.2 작동 원리: 문맥에 맞지 않으면 '0'으로 수렴 (Semantic Alignment)**

이 게이팅 메커니즘은 일종의 **노이즈 캔슬링** 역할을 합니다.

- **상황 A (정확한 정보):** 문맥이 "맛있는 빨간 OOO"이고, 검색된 N-gram이 "사과(Apple)"라면, Query와 Key의 유사도가 높아 게이트 값(alpha_t)이 1에 가까워집니다. → **정보 통과.**
- **상황 B (해시 충돌/다의어):** 문맥이 "아이폰 제조사 OOO"인데, 해시 충돌로 "사과(과일)" 정보가 검색되었다면, 문맥(Query)과 정보(Key)가 불일치하므로 게이트 값(alpha_t)은 0으로 수렴합니다. → **노이즈 차단.**

### **2.2.3 추가 정제: 경량 컨볼루션 (Lightweight Convolution)**

게이팅을 통과한 값(tilde{v}_t)에 대해 마지막으로 가벼운 **Depthwise Causal Convolution**을 한 번 더 수행합니다.

- **목적:** 단순히 값을 더하는 것을 넘어, 국소적인 수용 영역(Receptive field)을 조금 더 넓히고 비선형성(Non-linearity)을 추가하여 모델의 표현력을 높이기 위함입니다.

## 2.3 System Efficiency: Decoupling Compute and Memory

**비싸고 용량이 적은 GPU 메모리(HBM)의 한계를 아키텍처 레벨에서 극복한 방법**

→ **메모리 접근 주소가 입력 시점에 확정된다는 특성을 이용해, GPU가 연산하는 동안 CPU 메모리에서 데이터를 미리 가져오는 '비동기 프리패칭(Async Prefetching)'을 구현함으로써, GPU 메모리 용량의 물리적 한계를 소프트웨어적으로 돌파**

### **2.3.1 핵심 아이디어: "결정론적(Deterministic) 접근이 최적화를 낳는다"**

- **MoE와의 차이점:**
    - **MoE (기존):** 런타임에 계산된 Hidden State를 보고 라우터가 "어떤 전문가에게 보낼지" 결정합니다. 즉, **계산해보기 전에는 메모리 접근 패턴을 알 수 없습니다.**
    - **Engram (제안):** 입력 텍스트(Token Sequence)만 알면 어떤 N-gram 임베딩을 가져와야 할지 **계산 시작 전에 이미 100% 확정**됩니다.
- **의의:** 이 **'예측 가능성(Predictability)'** 덕분에 시스템 레벨에서 I/O 스케줄링을 짤 수 있게 됩니다.

### **2.3.2 추론(Inference) 최적화: Prefetching & Overlapping**

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_4.png)

가장 중요한 **"비동기 프리패칭(Asynchronous Prefetching)"** 전략입니다.

- **문제:** 수십~수백 기가바이트(GB)에 달하는 거대한 임베딩 테이블을 GPU HBM에 다 올릴 수 없습니다.
- **해결책 (PCIe 병목 숨기기):**
    1. 입력 토큰이 들어오면 CPU는 즉시 필요한 N-gram 인덱스를 계산합니다.
    2. GPU가 앞단 레이어(Transformer Block)를 열심히 연산하는 동안, CPU는 **백그라운드에서 호스트 메모리(DRAM)에 있는 임베딩 데이터를 PCIe 버스를 통해 GPU로 전송**합니다.
    3. Engram 레이어에 도달할 때쯤이면 데이터 전송이 끝나있습니다.
- **결과:** **통신 시간(Communication)을 연산 시간(Computation) 뒤로 숨기는(Overlapping)** 것이 가능해집니다. 실험 결과 100B 파라미터 테이블을 CPU로 오프로딩해도 오버헤드는 3% 미만입니다.

### **2.3.3 메모리 계층 구조 (Memory Hierarchy): Zipfian 법칙 활용**

데이터베이스 캐싱 전략과 동일한 원리를 적용합니다.

- **Zipfian 분포:** 자연어 데이터의 특성상, "자주 쓰이는 단어는 엄청 자주 쓰이고, 안 쓰이는 건 거의 안 쓰인다"는 롱테일(Long-tail) 법칙이 적용됩니다.
- **다계층 캐시 (Multi-Level Cache):**
    - **Hot Data (상위 빈도):** 빠른 **GPU HBM**이나 **Host DRAM**에 상주.
    - **Cold Data (롱테일 희귀 패턴):** 느리지만 용량이 큰 **NVMe SSD** 등에 저장.
- **효과:** 이를 통해 사실상 **무제한의 메모리 용량**을 모델에 제공하면서도, 실질적인 레이턴시는 최소화할 수 있습니다.

### **4. 학습(Training) 최적화: All-to-All 통신**

- 학습 시에는 여러 GPU에 임베딩 테이블을 쪼개서 저장(Sharding)합니다.
- **All-to-All 통신:** 각 GPU가 자신에게 필요한 행(Row)만 다른 GPU로부터 가져오고, 역전파(Backward) 때는 그래디언트를 다시 뿌려주는 표준적인 모델 병렬화(Model Parallelism) 기법을 사용하여, GPU 개수에 비례해 메모리 용량을 선형적으로 확장합니다.

# 3. Scaling Laws and Sparsity Allocation

---

**한정된 자원(파라미터 수)을 연산(MoE)과 기억(Engram) 중 어디에 투자해야 가장 효율적인가?**

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_5.png)

### **3.1 핵심 실험: "계산과 기억의 황금비율을 찾아라"**

연구진은 **총 파라미터(ISO-Parameter)**와 **연산량(ISO-FLOPs)**을 고정해 둔 상태에서, '비활성 파라미터(Sparse Parameter)' 예산을 MoE 전문가에게 줄지, Engram 메모리 슬롯에 줄지 비율을 조정하며 실험했습니다.

- **U자형 스케일링 법칙 (U-shaped Scaling Law):**
    - 실험 결과, 검증 손실(Validation Loss) 그래프가 뚜렷한 **U자 곡선**을 그렸습니다.
    - **순수 MoE:** 메모리 전용 모듈이 없어 정적 패턴을 연산으로 재구성하느라 비효율적입니다.
    - **Engram 과다:** 연산 능력이 부족해 동적인 추론을 못 합니다.
- **최적의 할당비 (Optimal Ratio):**
    - 약 **20~25%의 예산을 Engram(기억)에 할당**하고, 나머지 75~80%를 MoE(연산)에 할당했을 때 성능이 가장 좋았습니다.
    - 이는 MoE 전문가 수를 조금 줄이더라도, 그 자원으로 **전용 메모리(Lookup Table)를 확보하는 것이 전체 성능상 이득**이라는 것을 증명합니다.

### **2. 무한 메모리 영역 (Infinite Memory Regime)**

만약 파라미터 예산 제약을 풀고 메모리를 무한정 늘린다면 어떻게 될까요? 2.5절에서 다룬 '시스템 오프로딩' 덕분에 메모리 확장은 연산 비용(FLOPs) 증가를 유발하지 않습니다.

- **로그 선형 스케일링 (Log-Linear Scaling):**
    - 메모리 슬롯의 수를 늘릴수록 성능(Loss)은 **멱법칙(Power Law)**을 따르며 꾸준히 좋아집니다.
    - **의의:** 추가적인 연산 비용(GPU 태우기) 없이, 단순히 **메모리(RAM/SSD) 용량을 늘리는 것만으로도 모델 성능을 예측 가능하게 향상**시킬 수 있는 새로운 '튜닝 노브(Knob)'를 발견했습니다.

# 4. Large Scale Pre-training

**이론적으로 도출한 Engram 아키텍처와 할당 비율(Sparsity Allocation)을 실제 수백억 파라미터 규모의 모델에 적용하여 검증**

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_6.png)

### **4.1 실험 라인업: 공정한 비교를 위한 설계 (Fair Comparison)**

연구진은 총 2,620억(262B) 토큰으로 4가지 모델을 학습시켰습니다. 핵심은 **"학습 비용(FLOPs)과 파라미터 수를 통제한 상태"**에서 구조만 바꿨다는 점입니다.

- **Dense-4B:** 베이스라인 (4.1B 파라미터).
- **MoE-27B:** 표준적인 MoE 모델. 총 26.7B 파라미터지만, 실제 활성 파라미터는 Dense-4B와 유사합니다 (3.8B).
- **Engram-27B (핵심):** MoE-27B와 총 파라미터 수(26.7B)가 동일합니다. 단, **MoE의 전문가(Expert) 수를 72개에서 55개로 줄이고, 여기서 아낀 용량을 5.7B 크기의 Engram 메모리에 투자**했습니다.
- **Engram-40B (확장):** Engram-27B와 연산량은 똑같지만, 메모리 슬롯만 18.5B 규모로 대폭 늘린 모델입니다 (총 39.5B).

### **4.2 벤치마크 결과: 단순 암기를 넘어 '추론'을 잘한다**

실험 결과는 Engram의 압승입니다. 특히 개발자들이 주목해야 할 점은 **성능 향상의 '종류'**입니다.

- **지식(Knowledge) 태스크 향상:**
    - 예상대로 MMLU(+3.4), CMMLU(+4.0) 등 지식 집약적 태스크에서 점수가 올랐습니다. 메모리 모듈을 달았으니 당연한 결과입니다8.
- **추론(Reasoning) 및 코딩/수학의 비약적 향상 (Surprise!):**
    - 놀랍게도 메모리랑 상관없어 보이는 **일반 추론(BBH +5.0, ARC-Challenge +3.7)**과 **코딩/수학(HumanEval +3.0, MATH +2.4)**에서 더 큰 성능 향상폭을 보였습니다9.
    - **해석:** 이는 Engram이 단순 암기를 담당해준 덕분에, 모델의 **어텐션(Attention)과 MoE 전문가들이 복잡한 논리 연산에만 온전히 집중**할 수 있게 되었기 때문입니다 (Effective Depth 증가 효과).

# 5. Is Engram functionally equivalent to increasing the model’s depth?

**Engram이 단순한 지식 저장소가 아니라, 모델의 연산 효율을 높여 실질적으로 모델을 더 깊게(Deeper) 만드는 효과가 있는가?**

![image_7](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/conditional_memory_via_scalable_lookup/image_7.png)

### **1. 가설: "초반 레이어의 노가다를 없애주자"**

- **기존 LLM의 비효율:** "Diana, Princess of Wales"라는 개체를 인식하려면, 기존 모델은 "Diana" → "Princess" →  "of" → "Wales"라는 토큰들을 어텐션과 FFN을 통해 순차적으로 합치는 **특징 합성(Feature Composition)** 과정을 거쳐야 합니다.
- **Engram의 역할:** 이미 완성된 N-gram 임베딩("Diana, Princess of Wales")을 O(1) 조회로 던져줍니다. 즉, 모델이 초반 레이어에서 수행하던 **'단어 조립 노가다'를 건너뛰게(Bypass)** 해줍니다.

### **2. 검증 1: LogitLens (예측 수렴 속도)**

- **LogitLens란?** 중간 레이어의 히든 스테이트를 바로 출력층(LM Head)에 꽂았을 때, 다음 토큰을 얼마나 잘 맞추는지(KL Divergence)를 측정하는 도구입니다.
- **결과:** Engram 모델은 MoE 베이스라인보다 **훨씬 더 이른 레이어에서 KL Divergence가 급격히 떨어집니다**.
- **의미:** 모델이 정답을 확신하는 시점이 앞당겨졌습니다. 즉, 특징 추출과 개념 조립이 훨씬 빨리 끝났다는 뜻입니다.

### **3. 검증 2: CKA (레이어 유사도 분석)**

- **CKA(Centered Kernel Alignment)란?** 두 신경망의 내부 표현(Representation)이 얼마나 유사한지를 측정하는 지표입니다.
- **결과 (Effective Depth):**
    - Engram 모델의 **Layer 5**의 표현력은 MoE 베이스라인의 **Layer 12**와 가장 유사했습니다.
    - 유사도 히트맵에서 대각선이 위쪽으로 이동(Upward shift)하는 현상이 뚜렷하게 관찰됩니다.
- **의미:** Engram을 장착하면 **약 7개 레이어 분량의 연산 깊이를 '공짜로' 얻는 효과**가 있습니다. 초반 5개 레이어만으로도 기존 모델 12개 레이어 수준의 추상화된 정보를 처리하고 있다는 뜻입니다.
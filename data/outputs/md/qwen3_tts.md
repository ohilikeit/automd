# 1. Introduction

---

![image_1](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_1.png)

**Qwen3-TTS**는 Qwen 시리즈의 첫 번째 텍스트-음성 변환(TTS) 모델로, 5백만 시간 이상의 음성 데이터로 학습된 **다국어, 제어 가능(Controllable), 고성능 스트리밍** 모델입니다.

### 주요 특징

- **제어 가능성 (Controllability):** 자연어 설명을 통해 새로운 목소리를 생성하거나 생성된 음성의 세부 속성을 정밀하게 제어할 수 있습니다.
- **음성 복제 (Voice Cloning):** 3초 분량의 참조 오디오만으로도 고품질의 제로샷(Zero-shot) 음성 복제가 가능합니다.
- **자연스러움 (Naturalness):** 1.7B 모델은 사람과 매우 유사한 수준(SOTA)의 음성 품질을 제공합니다.
- **다국어 지원 (Multilinguality):** 10개 이상의 언어를 지원하며, 언어 간 일관된 화자 생성이 가능합니다.
- **스트리밍 (Streaming):** 텍스트 입력과 동시에 오디오를 출력하는 스트리밍 설계를 통해, 0.6B 모델 기준 **97ms**라는 극도로 낮은 첫 패킷 지연(First-packet latency)을 달성했습니다.

### 아키텍처 혁신: 두 가지 토크나이저

LLM과의 원활한 통합과 초저지연을 위해 두 가지 유형의 토크나이저를 도입했습니다.

| **특징** | **단일 코드북 (25Hz 모델)** | **다중 코드북 (12Hz 모델)** |
| --- | --- | --- |
| **데이터 구조** | 시간당 1개의 정수 (`[T]`) | 시간당 N개의 정수 벡터 (`[T, N]`) |
| **정보 저장** | 내용+음향 뭉뚱그려 압축 | Layer 0: 내용(Semantic) 
 Layer 1~N: 음향 디테일(Acoustic)  |
| **복원 방식** | **DiT (Diffusion)**: 뭉뚱그린 정보에서 디테일을 '생성'해내야 함  | **MTP + ConvNet**: 나눠담은 정보를 합치기만 하면 됨 (빠름)  |
| **주요 목적** | 고품질 생성, 복잡한 추론 | **초저지연 스트리밍**, 실시간 처리  |
- **Qwen-TTS-Tokenizer-25Hz:** 의미와 음향 정보를 통합한 단일 코드북 방식입니다. **Flow Matching**을 통한 파형 재구성을 지원하며 스트리밍과 고품질 생성의 균형을 맞춥니다.
- **Qwen-TTS-Tokenizer-12Hz:** **12.5Hz**의 다중 코드북(Multi-codebook) 방식입니다. 확산 모델 없이 경량화된 Causal ConvNet만으로 파형을 복원하며, **MTP(Multi-Token Prediction)** 모듈을 통해 첫 프레임부터 즉각적인 음성 디코딩이 가능하도록 설계되어 초저지연 응용에 최적화되었습니다.

### 성능 성과

- **SOTA 달성:** Seed-TTS 벤치마크에서 가장 낮은 단어 오류율(WER)을 기록하며 제로샷 음성 복제 분야의 새로운 기준을 세웠습니다.
- **상용 모델 상회:** MiniMax, ElevenLabs와 같은 상용 모델 대비 10개 언어 모두에서 더 높은 화자 유사도를 보였습니다.
- **장문 생성:** 10분 이상의 긴 음성도 끊김 없이 자연스럽게 생성할 수 있는 안정성을 확보했습니다.

![image_2](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_2.png)

# 2. Qwen-TTS-Tokenizer

---

## 2.1 Qwen-TTS-Tokenizer-25Hz (고품질 & 스트리밍 균형)

- **기반 및 구조:** Qwen2-Audio를 기반으로 한 **25Hz 단일 코드북(Single-codebook)** 토크나이저입니다.
- **2단계 학습 프레임워크:**
    - **1단계 (Stage 1):** ASR(자동 음성 인식) 태스크를 통해 Qwen2-Audio를 계속 사전 학습시키며, 중간에 VQ(Vector Quantization) 레이어를 추가합니다.
    - **2단계 (Stage 2):** 멜-스펙트로그램 디코더를 통합하여 전체 모델을 미세 조정(Fine-tuning)함으로써 필수적인 음향 정보를 토큰에 주입합니다.
- **스트리밍 디토크나이저 (Detokenizer):**
    - **DiT & Flow Matching:** Flow Matching으로 훈련된 Diffusion Transformer(DiT)를 사용하여 코드를 멜-스펙트로그램으로 매핑하고, BigVGAN이 이를 파형으로 복원합니다.
    - **슬라이딩 윈도우 어텐션:** 스트리밍을 지원하기 위해 수신 필드(Receptive field)를 4개의 블록(현재, 과거 3개, 미래 1개)으로 제한하여 청크 단위로 멜-스펙트로그램을 생성합니다.

## 2.2 Qwen-TTS-Tokenizer-12Hz (초저지연 & 효율성 특화)

![image_3](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_3.png)

- **구조 및 특징:** 의미(Semantic)와 음향(Acoustic) 스트림이 공동 최적화된 **12.5Hz 멀티 코드북(Multi-codebook)** 토크나이저입니다.
- **정보 분리 및 압축 (Disentanglement):**
    - **의미 경로 (Semantic):** 첫 번째 코드북 레이어는 WavLM(소리의 특징을 잘 파악하는 모델)을 교사(Teacher) 모델로 사용하여 고수준의 의미 내용을 캡처합니다.
    - **음향 경로 (Acoustic):** 이후 레이어들은 15-layer RVQ(Residual Vector Quantization)를 통해 세밀한 음향 디테일과 운율을 캡처합니다.
- **완전 인과적 설계 (Fully Causal Design):**
    - 룩어헤드(Look-ahead)가 없는 완전 인과적 인코더/디코더를 사용하여, 토큰이 생성되는 즉시 오디오를 점진적으로 재구성할 수 있습니다.
    - 이는 복잡한 확산 모델 없이 경량화된 Causal ConvNet을 사용해 실시간 온라인 서비스에 적합한 초저지연 성능을 보장합니다.

# 3. Method

---

## 3.1 아키텍처

![image_4](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_4.png)

Qwen3-TTS는 높은 동시성 처리와 낮은 지연 시간을 위해 **Qwen3 LM**을 백본으로 사용하며, 텍스트와 음성 토큰을 결합한 **이중 트랙(Dual-track)** 방식(동시에 진행)을 채택했습니다.

**A. Qwen3-TTS-25Hz 모델 (고품질용)**

- **백본의 역할:** 텍스트와 이전 음성 정보를 바탕으로, **현재 시점의 단일 음성 토큰**을 예측합니다.
- **연결:** 예측된 값은 `Linear Head`를 통과하여 토큰이 됩니다.
- **출력:** 이 토큰들이 모이면 **Chunk-wise DiT (Diffusion Transformer)** 모듈로 보내져서 고품질 파형으로 바뀝니다.

**B. Qwen3-TTS-12Hz 모델 (초저지연용)**

- **백본의 역할:** 여기서는 백본이 **0번째 코드북(의미 정보)** 하나만 예측합니다. (나머지 15개 디테일은 신경 쓰지 않음)
- **MTP 모듈 (핵심):** 백본이 "의미"를 던져주면, **MTP(Multi-Token Prediction) 모듈**이 이어받아 나머지 1~15번 코드북(음향 디테일)을 순식간에 예측해냅니다.
- **출력:** 이렇게 완성된 16개 토큰 세트는 가벼운 **Code2Wav** 모듈을 통해 즉시 소리로 바뀝니다.

## 3.2 학습 과정

모든 데이터는 제어 가능한 음성 생성을 위해 **ChatML 포맷**으로 표준화되어 학습됩니다.

- **사전 학습 (Pre-training) - 3단계:**
    1. **일반 단계 (General Stage):** 500만 시간 이상의 다국어 데이터로 일반적인 텍스트-음성 매핑 학습.
    2. **고품질 단계 (High-Quality Stage):** 고품질 데이터로 환각(Hallucination) 현상 제거 및 품질 향상.
    3. **장문 단계 (Long-Context Stage):** 최대 토큰 길이를 **32,768**로 늘리고 긴 음성 데이터를 업샘플링하여 긴 문맥 처리 능력 강화.
- **사후 학습 (Post-training):**
    - **DPO (Direct Preference Optimization):** 인간 피드백을 반영하여 모델 정렬.
    - **SFT (Speaker Fine-tuning):** 특정 화자의 목소리, 자연스러움, 제어 능력을 위한 경량 미세 조정.

## 3.3 주요 기능

- **음성 복제:** 화자 임베딩(실시간) 또는 인컨텍스트 러닝(운율 보존) 방식을 통해 3초 분량의 데이터로 복제 가능.
- **음성 디자인 (Thinking Pattern):** 복잡한 설명 지침을 더 잘 따르기 위해, 학습 중에 확률적으로 활성화되는 **생각하는 패턴(Thinking pattern)**을 도입하여 추론 능력을 강화했습니다.

# **4. Experiments**

---

## 4.1 토크나이저 성능

![image_5](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_5.png)

- **12Hz 토크나이저의 혁신:** Qwen-TTS-Tokenizer-12Hz는 기존의 Mimi나 SpeechTokenizer 등을 제치고 음성 재구성 품질(PESQ, STOI 등)에서 SOTA를 기록했습니다. 이는 극도로 낮은 비트레이트와 효율성을 유지하면서도 최고의 음질을 달성했다는 점에서 의미가 큽니다.

## 4.2 제로샷 음성 생성

![image_6](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_6.png)

- **최고의 정확도:** Seed-TTS 벤치마크에서 **가장 낮은 단어 오류율(WER)**을 기록하며 CosyVoice 3, Seed-TTS 등의 경쟁 모델을 앞섰습니다.
- **12Hz vs 25Hz:** 흥미롭게도 콘텐츠 정확도(WER) 면에서는 **12Hz 모델이 25Hz 모델보다 더 우수한 성능**을 보였습니다. 이는 12Hz의 낮은 시간 해상도가 장기적인 의존성(Long-term dependencies)을 모델링하는 데 더 유리하기 때문인 것으로 분석됩니다.

## 4.3 다국어 및 교차 언어 능력

![image_7](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_7.png)

- **상용 모델 압도:** MiniMax-Speech, ElevenLabs와 비교했을 때, 10개 언어 **모두에서 더 높은 화자 유사도(Speaker Similarity)**를 기록했습니다. 이는 Qwen3-TTS가 언어가 바뀌어도 화자의 고유한 음색을 가장 잘 유지한다는 것을 의미합니다.
- **중국어-한국어(Zh-to-Ko) 성능:** 교차 언어 평가에서 특히 중국어를 한국어로 변환할 때, 기존 CosyVoice3 대비 **오류율을 약 66% 감소**시키는 놀라운 성능 향상을 보였습니다.

## 4.4 제어 가능한 생성

![image_8](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/qwen3_tts/image_8.png)

- 감정을 포함하여 목소리의 나이, 성별, 직업적 톤, 말하는 속도 등을 채팅하듯이 말로 다 바꿀 수 있는 기능
- **지시 이행 능력:** 목소리를 수정(Editing)하는 작업에서 **GPT-4o-mini-tts보다 뛰어난 성능**을 보였습니다.
- **Voice Design:** 텍스트 설명으로 새로운 목소리를 만드는 작업에서 오픈소스 모델 중 SOTA를 달성했습니다.
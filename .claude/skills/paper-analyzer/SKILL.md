---
name: paper-analyzer
description: "AI/ML 논문 PDF를 분석하여 팀 내부 공유용 한국어 마크다운 문서를 생성하는 skill. 논문의 핵심 내용(개요, 구조/방법론, 실험 결과)을 중심으로, 논문 내 figure를 추출하고, 친절한 한국어 설명과 함께 가독성 높은 md 파일을 만든다. data/pdf/ 디렉토리에 PDF가 있거나 논문 분석/요약/정리를 요청할 때 반드시 이 skill을 사용한다. 논문 리뷰, paper review, 논문 요약, 논문 정리, PDF 분석, 논문 읽기, paper summary 등의 키워드에도 반응한다."
user-invocable: true
argument-hint: <pdf_filename>
---

# Paper Analyzer: 논문 분석 → 한국어 마크다운 문서 생성

AI/ML 논문 PDF를 읽고, 핵심 내용을 추출하여 팀 내부 공유용 한국어 마크다운 문서를 만든다.

분석할 논문: $ARGUMENTS

## 프로젝트 구조

```
data/
├── pdf/                          # 분석 대상 논문 PDF
├── outputs/
│   ├── images/{paper_name}/      # 논문에서 추출한 figure 이미지
│   ├── md/{paper_name}.md        # 최종 결과물 (GitHub 배포용, raw URL)
│   └── md/{paper_name}_local.md  # 최종 결과물 (로컬용, 상대 경로)
```

- **paper_name**: PDF 파일명에서 확장자를 제거 (`heterogeneous.pdf` → `heterogeneous`)
- **GitHub 이미지 URL**: `https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/{paper_name}/image_N.png`
- **로컬 이미지 경로**: `../images/{paper_name}/image_N.png`
- **Figure 추출 스크립트**: `${CLAUDE_SKILL_DIR}/scripts/extract_figures.py`

---

## 핵심 원칙 (먼저 읽기)

이 skill은 과거의 실패 경험을 반영해 다음 원칙을 따른다.

1. **Figure는 캡션과 함께 한 단위로 추출한다.** 캡션을 잘라내려 하지 않는다. figure 박스 + 캡션이 시각적으로 자연스러운 사각형이며, figure 식별도 명확하다.
2. **자동 검출(auto mode)을 우선 사용한다.** 캡션 패턴(`Figure N:`, `Table N:`, `Figure N <대문자>`)을 인식해서 그 위/아래 그래픽 영역을 자동으로 묶어준다. bbox 좌표를 직접 추정하는 일은 fallback이다.
3. **단일 컬럼 figure도 컬럼 폭 + 캡션 단위로 통째로 잡는다.** figure 박스만 좁게 도려내면 좌우 여백이 어색해지고 캡션이 잘려 정보 손실이 생긴다.
4. **검증은 잘림 없음 + 미적 기준 둘 다 통과해야 한다.** "기능적으로 보이기만 하면 OK"가 아니라 "사용자가 봤을 때 깔끔한가"까지 점검한다.

---

## 전체 워크플로우 (5단계)

### Phase 1: 논문 읽기

`Read` 도구로 PDF를 읽으며 다음을 파악한다.

- 이 논문이 해결하려는 **문제**
- 제안 방법의 **핵심 아이디어** (한 문장으로 압축 가능한가?)
- 가장 인상적인 **실험 결과**
- 논문 내 **Figure/Table 목록**과 각각의 역할

긴 논문(>30페이지)은 Introduction → Method → Experiments 순으로 우선 읽고 부록은 필요시 참조한다.

### Phase 2: 문서 구조 설계

기존 예시(`data/outputs/md/` 내 파일)를 1~2개 읽어 톤을 상기한 뒤, 다음 패턴으로 뼈대를 짠다.

```
# 1. Introduction
---
(배경/문제 → 제안 방법 → 핵심 결과 요약 — figure 1~2개 사용)

# 2. [핵심 방법론 섹션]
---
(아키텍처/알고리즘/메커니즘 — figure 1~2개)

# 3. [추가 핵심 섹션] (선택)
---

# N. Experiments
---
(실험 설정 → 메인 결과 figure 해석 → 인사이트)
```

- Related Work는 포함하지 않는다. 배경이 필요하면 Introduction에 녹인다.
- 논문 모든 섹션을 1:1 옮기지 않고 **핵심 기여만** 선별한다.
- 섹션 수는 보통 3~6개.

### Phase 3: Figure 추출 (auto mode 우선)

#### 3-1. PDF 정보 확인 + figure 자동 검출

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> <out_dir> --mode info
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> <out_dir> --mode detect
```

`detect`는 이미지를 저장하지 않고 발견된 모든 Figure/Table 후보의 라벨, 페이지, bbox, 캡션 첫 200자를 JSON으로 출력한다. **본문에 사용할 figure를 이 목록에서 골라 라벨을 기억한다.**

#### 3-2. 자동 추출 (가장 권장)

본문에 사용할 figure만 골라서 한 번에 자동 추출한다.

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> data/outputs/images/{paper_name}/ \
  --mode auto \
  --select "Figure 1,Figure 2,Figure 3,Figure 4,Table 1,Figure 5,Figure 6"
```

- `--select`로 명시한 순서가 `image_1.png`, `image_2.png`, ... 순서가 된다.
- `--select` 생략 시 모든 검출 결과가 추출된다 (부록 figure까지 포함되니 보통 명시한다).
- 출력 JSON에서 각 항목의 `label` 필드를 보고 마크다운에서 어떤 figure인지 매핑한다.

자동 검출 알고리즘은 다음을 수행한다:
- **캡션 위치**(Figure는 그림 아래 / Table은 표 위)에서부터 검색 방향 결정
- **그래픽 + 텍스트 영역**을 캡션과 인접한 것끼리 묶기
- **본문 단락 차단**: 캡션과 그래픽 사이에 본문 텍스트(폭 80pt+, 길이 12pt+)가 있으면 분리
- **abstract/definition 박스 차단**: 그래픽 안의 텍스트 비율이 50% 이상이면 본문 컨테이너로 판단
- **컬럼 인식**: 1-column / 2-column 자동 감지하여 단일 컬럼 figure는 컬럼 폭으로 정규화

#### 3-3. 검증 (필수, 생략 불가)

추출된 모든 이미지를 `Read`(vision)로 직접 열어서 점검한다.

**합격 기준 — 5가지 모두 통과해야 함:**

- [ ] figure 본체가 잘림 없이 포함되었나
- [ ] **캡션이 포함되어 있나** (figure 식별 + 시각적 균형 확보)
- [ ] 다른 figure/박스/본문 텍스트가 섞이지 않았나
- [ ] **좌우 여백이 균형 있나** (단일 컬럼 figure가 컬럼 폭에 맞게 잡혔나)
- [ ] **비율이 자연스러운가** (지나치게 좁거나 길지 않은가)

**불합격이면 fallback으로 수동 crop**(3-4)을 시도한다. auto 모드가 같은 figure를 두 번 시도해도 같은 결과를 내므로, 수동 조정만이 의미 있다.

#### 3-4. Fallback: 수동 crop (auto 검출이 실패한 경우만)

벡터 그림이 너무 복잡하거나 캡션이 없는 figure는 자동 검출이 실패할 수 있다. 이 경우에만 수동 좌표를 사용한다.

먼저 페이지를 통째로 렌더링해서 위치를 눈으로 확인:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> /tmp/{paper_name}_pages \
  --mode page --pages 2,5,8
```

PDF 좌표계는 **좌측 상단 원점**, 표준 페이지는 **612×792 pt**. `Read`로 페이지 이미지를 보고 figure가 있는 영역의 bbox를 추정한다.

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> data/outputs/images/{paper_name}/ \
  --mode crop \
  --regions '[{"page":3,"bbox":[50,50,560,420]}]' \
  --start-index 3   # image_3.png부터 저장
```

**bbox 추정 가이드:**
- 본문 영역은 보통 `x: 50~560`, `y: 50~750`
- 단일 컬럼 figure (좌측 컬럼): `x ≈ 50~300` / 우측 컬럼: `x ≈ 310~560`
- **캡션은 의도적으로 포함**시킨다 — y1을 캡션 끝까지 잡는다
- 페이지 헤더(예: 논문 제목)가 `y < 50`에 있으니 `y0 ≥ 50`

PyMuPDF의 좌표 처리에 미세 오프셋이 있을 수 있으므로, 한 번 추출 → vision 검증 → 미세 조정의 사이클을 따른다 (figure 1개당 최대 3회).

#### 3-5. 임베디드 raster 이미지 스캔 (참고용)

논문에 본문 직접 임베디드 raster 이미지가 있는지 빠르게 확인하려면:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf> /tmp/{paper_name}_scan --mode scan
```

대부분의 학술 논문은 벡터 그래픽이라 이 모드는 보조 정보만 제공한다.

### Phase 4: 마크다운 작성

Phase 2의 구조와 Phase 3의 이미지를 결합한다. **로컬 상대 경로**(`../images/{paper_name}/image_N.png`)로 작성하고, GitHub URL 변환은 Phase 5에서 한다.

#### 글쓰기 원칙

1. **한국어 작성**, 기술 용어(Attention, MoE, FLOP 등)는 영어 원문 유지.
2. **ML 전문가 독자 가정**, 단 해당 논문의 고유 개념은 **친절하게 풀어서** 설명한다.
   - 좋음: "Engram이 초반 레이어에서 정적 지식 재구성을 전담함으로써 깊은 레이어들이 복잡한 추론에 집중할 수 있게 합니다."
   - 나쁨: "Engram은 conditional memory module이다." (용어만 던지기)
3. 핵심이 아닌 내용은 과감히 생략한다. **컴팩트함**이 생명.
4. 핵심 개념엔 **비유나 직관적 설명**을 곁들인다.
   - 예: "이는 마치 정적인 조회 테이블을 런타임에 값비싼 연산으로 재구성하는 것과 같습니다"
   - 예: "→ **트럼프 뒤에 대통령이 올 확률을 계산하기 위해 수많은 뉴런을 쓰는 건 낭비입니다.**"

#### 포맷 규칙

1. 최상위 제목(`#`) 직후 `---` 수평선
2. 이미지 삽입: `![image_1](../images/{paper_name}/image_1.png)`
3. 핵심 개념/결론은 `**볼드**`
4. 구조화된 정보는 불릿 리스트 (들여쓰기로 계층)
5. 수식은 LaTeX (`$...$` / `$$...$$`)
6. 비교는 마크다운 테이블
7. 핵심 인사이트는 `> ...` 인용 블록
8. 한 줄 요약 패턴: `→ **핵심 요약**`

#### Figure 해석 원칙 (가장 중요)

논문 figure를 삽입하면 **반드시 상세한 한국어 해석**을 함께 쓴다. 이것이 이 문서의 핵심 가치다.

1. **그래프/차트**: 축의 의미, 각 선/막대의 의미, 핵심 트렌드를 설명.
   - 예: "가로축은 참여 에이전트 수(N=4, 8, 16)입니다. 그룹 인원이 많아질수록 초록 막대가 점점 낮아지는 것을 볼 수 있습니다."
2. **아키텍처 다이어그램**: 각 구성요소의 역할과 데이터 흐름을 순서대로.
   - 예: "1. Broadcast: 에이전트가 자신의 제안과 논리를 다른 모든 에이전트에 전파합니다."
3. **비교 실험**: 각 조건의 결과를 대비시키며 **"왜 이런 차이가 나는지"**까지 해석.
4. "Figure 3은 성능 비교다"라고만 쓰지 않는다. 독자가 figure를 보지 않아도 내용을 이해할 수 있을 만큼 상세하게.

#### Introduction 패턴

```markdown
# 1. Introduction

---

### 1. 배경 및 문제 제기
- 분야의 현재 상황과 한계점
- 기존 접근법의 구체적 문제

### 2. 제안: [방법 이름]
- 핵심 아이디어를 직관적으로
- 기존 방법과의 차별점

### 3. 핵심 결과
- 가장 인상적인 수치
- 실용적 의의
```

#### Experiments 패턴

수치 나열이 아니라 **해석과 인사이트 중심**.
- 실험 설정을 간결하게
- 결과 figure를 삽입 + 상세 해석
- **"왜 이런 결과가 나왔는가"** 분석 포함
- 핵심 발견은 볼드 또는 💡 마크

#### 반복 개선 (3회)

각 회차마다 기존 예시 1개를 다시 읽고 톤을 상기한 뒤 검토한다.

```
1회차: 초안 작성 → 기존 예시와 구조/포맷 비교
2회차: 설명 깊이/친절함 개선 → 이미지-텍스트 매핑 점검
3회차: 컴팩트함, 일관성, 오탈자 점검
```

### Phase 5: 두 개 파일 생성 (로컬용 + GitHub용)

| 파일 | 이미지 경로 | 용도 |
|---|---|---|
| `data/outputs/md/{paper_name}_local.md` | `../images/{paper_name}/image_N.png` | 로컬 미리보기 |
| `data/outputs/md/{paper_name}.md` | `https://raw.githubusercontent.com/.../image_N.png` | GitHub 배포 |

```bash
cp data/outputs/md/{paper_name}_local.md data/outputs/md/{paper_name}.md
sed -i "s|\.\./images/${PAPER_NAME}/|https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/${PAPER_NAME}/|g" data/outputs/md/{paper_name}.md
```

두 파일이 모두 존재하고 이미지 링크 패턴이 올바른지 마지막으로 확인한다.

---

## 흔한 실패 모드 (이걸 피해라)

### 1. 캡션을 잘라내려 하기
캡션을 figure 박스 외부로 보고 잘라내면 figure 폭이 어색해지고 식별이 어려워진다. 항상 캡션을 포함시켜라.

### 2. 단일 컬럼 figure를 그림 박스 폭으로만 자르기
페이지 우측 컬럼에만 있는 figure도 컬럼 폭에 맞춰 캡션과 함께 정사각형/직사각형 비율로 잡아야 자연스럽다.

### 3. 좌표를 미세하게 5pt씩 조정하며 4~5번 반복
auto 모드를 먼저 시도하고, 그래도 안 되면 수동 crop은 한 번에 넉넉히 잡아 캡션까지 포함하라. 미세 조정 루프는 시간 낭비.

### 4. "잘림 없음"만 체크하고 합격 처리
잘림 없어도 좌우 여백이 들쭉날쭉하거나 비율이 어색하면 불합격. 미적 기준도 함께 본다.

### 5. PDF 좌표계 변환을 이론값만 믿기
실제 추출 스크립트는 약간의 시스템 오프셋을 가질 수 있다. 첫 시도 결과를 vision으로 보고 실제 오프셋을 역산하라.

---

## 주의사항

- 긴 논문(>30페이지): 본문(Introduction~Experiments) 우선, 부록은 필요시.
- 논문마다 구조가 다르다. 위 패턴을 기계적으로 적용하지 말고 **핵심 기여에 맞게 유연하게** 조정한다.
- 이미지 파일명은 반드시 `image_1.png`, `image_2.png`, ... 순서.
- 추출 디렉토리는 `data/outputs/images/{paper_name}/`. 없으면 자동 생성.

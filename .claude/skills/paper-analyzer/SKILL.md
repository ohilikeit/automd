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
│   └── md/{paper_name}.md        # 최종 결과물: 논문 분석 마크다운 (GitHub용)
│   └── md/{paper_name}_local.md  # 최종 결과물: 논문 분석 마크다운 (로컬용)
```

- **GitHub 이미지 URL 패턴**: `https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/{paper_name}/image_N.png`
- **로컬 이미지 상대 경로**: `../images/{paper_name}/image_N.png`
- **paper_name**: PDF 파일명에서 확장자를 제거한 값 (예: `attention_is_all_you_need.pdf` → `attention_is_all_you_need`)
- **Figure 추출 스크립트**: `${CLAUDE_SKILL_DIR}/scripts/extract_figures.py`

---

## 전체 워크플로우

아래 6단계를 순서대로 수행한다. 특히 **3회 이상의 반복 개선**이 핵심이다.

### Phase 1: 논문 깊이 읽기

PDF를 처음부터 끝까지 읽되, 단순 요약이 아니라 **논문의 핵심 기여가 무엇인지** 파악하는 데 집중한다.

1. `Read` 도구로 PDF 전체를 읽는다 (큰 논문은 페이지 범위를 나눠서).
2. 읽으면서 아래를 파악한다:
   - **이 논문이 해결하려는 문제**는 무엇인가?
   - **제안하는 방법**의 핵심 아이디어는?
   - **실험 결과**에서 가장 인상적인 발견은?
   - 기존 방법 대비 **차별점**은?
   - 논문 내 **Figure/Table 목록**과 각각이 어떤 내용을 설명하는지

이 단계에서 충분한 시간을 들여 깊이 이해해야 한다. 이해가 부족하면 이후 단계의 품질이 떨어진다.

### Phase 2: 문서 구조 설계

기존 예시 문서들의 패턴을 따라 전체 문서의 뼈대를 설계한다.

**필수 구조 패턴:**
```
# 1. Introduction
---
(배경/문제 → 제안 방법 → 핵심 결과 요약)

# 2. [핵심 방법론/구조 섹션]
---
(아키텍처, 알고리즘, 핵심 메커니즘 상세 설명)
(필요시 하위 섹션으로 분리)

# 3. [추가 핵심 섹션] (논문에 따라)
---
(스케일링 법칙, 학습 방법, 특수 기능 등)

# N. Experiments
---
(실험 설정, 결과 해석, 핵심 발견)
```

구조를 설계할 때 고려할 점:
- **이전 연구(Related Work)는 포함하지 않는다.** 배경 설명이 필요하면 Introduction에 간결하게 녹인다.
- 논문의 모든 섹션을 1:1로 옮기지 않는다. **핵심 기여에 해당하는 내용만** 선별한다.
- 각 섹션에서 어떤 figure를 사용할지 미리 매핑한다.
- 섹션 수는 보통 3~6개가 적절하다.

### Phase 3: Figure 추출

논문에서 필요한 figure들을 이미지로 추출한다. 이 단계가 매우 중요하다.

**3-1. PDF 정보 확인:**
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf_path> <output_dir> --mode info
```

**3-2. 임베디드 이미지 탐색 (먼저 시도):**
일부 논문에는 래스터 이미지가 직접 임베디드되어 있다. 먼저 scan 모드로 확인한다.
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf_path> /tmp/{paper_name}_scan --mode scan
```
- 결과가 있으면 `/tmp/{paper_name}_scan/` 내 이미지를 vision으로 확인하여 쓸 만한 것을 선별한다.
- 결과가 0개이면 (벡터 그래픽 기반 논문) 아래 3-3 ~ 3-5의 page+crop 방식을 사용한다.

**3-3. Figure 위치 파악 (page 렌더링):**
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf_path> /tmp/{paper_name}_pages --mode page --pages 2,3,4,5,6
```
- 렌더링된 페이지 이미지를 `Read` 도구(vision)로 열어서 각 figure의 위치를 눈으로 확인한다.
- figure가 있는 페이지 번호와 페이지 내 대략적인 위치(상단/중간/하단)를 기록한다.

**3-4. Figure 영역 정밀 캡쳐 (crop):**

PDF 좌표계 기준으로 bbox `[x0, y0, x1, y1]`을 지정한다. 표준 논문 페이지는 **612 x 792 pt**이다.

> **bbox 추정 실전 가이드:**
> - 좌표 원점은 **페이지 좌측 상단**이다 (x→오른쪽, y→아래쪽).
> - 대부분의 논문 본문 영역: x는 `50~560` 범위, y는 `50~750` 범위.
> - **figure가 페이지 상단 1/3에 위치**: bbox ≈ `[50, 50, 560, 300]`
> - **figure가 페이지 상단 1/2에 위치**: bbox ≈ `[50, 50, 560, 420]`
> - **figure가 페이지 중앙**: bbox ≈ `[50, 200, 560, 550]`
> - **figure가 페이지 하단 1/2**: bbox ≈ `[50, 400, 560, 750]`
> - 2-column 논문에서 figure가 전체 폭을 차지: x를 `50~560`으로.
> - 2-column 논문에서 한쪽 컬럼에만 있는 figure: 왼쪽 `[50, y0, 300, y1]`, 오른쪽 `[310, y0, 560, y1]`
> - **figure 캡션은 제외**하는 것이 좋다. 설명은 md 파일에서 한국어로 작성한다.
> - **"Preprint" 같은 헤더 텍스트**가 포함되지 않도록 y0을 충분히 아래로 잡는다 (보통 `y0 ≥ 55`).

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/extract_figures.py <pdf_path> data/outputs/images/{paper_name}/ \
  --mode crop \
  --regions '[{"page": 3, "bbox": [50, 55, 560, 300]}, {"page": 5, "bbox": [50, 55, 560, 420]}]' \
  --dpi 250
```

**3-5. 추출 결과 검증 및 반복 조정 (필수 — 생략 금지):**

> **이 단계는 절대 건너뛸 수 없다.** 모든 추출된 이미지를 하나하나 `Read` 도구(vision)로 열어서 직접 눈으로 확인해야 한다. vision 확인 없이 다음 단계로 넘어가는 것은 금지한다.

**검증 체크리스트 (이미지 1장마다 반드시 수행):**

1. `Read` 도구로 `data/outputs/images/{paper_name}/image_N.png`를 연다.
2. 아래 항목을 하나씩 확인한다:
   - [ ] figure 전체가 잘림 없이 포함되어 있는가?
   - [ ] 불필요한 텍스트("Preprint" 헤더, 본문 텍스트, 페이지 번호)가 섞여있지 않은가?
   - [ ] figure 캡션이 제외되어 있는가? (캡션은 md에서 한국어로 작성)
   - [ ] 텍스트/라벨이 선명하게 읽히는가?
   - [ ] 여백이 과하지 않은가?
3. **하나라도 불합격이면** bbox를 조정하여 재캡쳐하고, 다시 vision으로 확인한다.
4. figure 하나당 **최대 3회까지 조정**을 시도한다.

**불합격 시 조정 가이드:**
- **잘린 경우**: y1(하단) 또는 x1(우측)을 20~50pt 늘린다.
- **여백이 과한 경우**: 해당 방향의 좌표를 10~30pt 좁힌다.
- **"Preprint" 헤더 포함**: y0을 60~70으로 올린다.
- **흐림/저해상도**: `--dpi 300`으로 재캡쳐한다.

모든 이미지가 합격한 뒤에야 Phase 4로 진행한다.

### Phase 4: 마크다운 문서 작성

Phase 2의 구조와 Phase 3의 이미지를 결합하여 최종 md 파일을 작성한다.

**작성 시에는 로컬 상대 경로**(`../images/{paper_name}/image_N.png`)를 사용한다. GitHub URL 변환은 Phase 6에서 수행한다.

**반드시 따라야 하는 작성 규칙:**

#### 글쓰기 원칙

1. **한국어로 작성**하되, 기술 용어(Attention, MoE, FLOP 등)는 영어 원문을 유지한다.
2. **ML 전문가가 읽는다고 가정**하되, 해당 논문의 고유한 개념/기법은 친절하게 풀어 설명한다.
   - 좋은 예: "Engram이 초반 레이어에서 정적인 지식 재구성을 전담함으로써, 모델의 **깊은 레이어들이 복잡한 추론에 집중할 수 있게 합니다.**"
   - 나쁜 예: "Engram은 conditional memory module이다." (설명 없이 용어만 던지기)
3. 핵심이 아닌 내용은 과감히 생략한다. **컴팩트함**이 생명이다.
4. 단, 핵심 개념 중 이해에 도움이 되는 부분은 **비유나 직관적 설명**을 덧붙인다.
   - 예: "이는 마치 정적인 조회 테이블(Lookup table)을 런타임에 값비싼 연산으로 재구성하는 것과 같으며"
   - 예: "→ **트럼프 라는 단어 뒤에 대통령이 올 확률을 계산하기 위해 수많은 뉴런을 거치는 것은 낭비다.**"

#### 포맷 규칙

1. **섹션 구분**: 최상위 제목(`#`) 바로 아래에 `---` 수평선을 넣는다.
2. **이미지 삽입**: 로컬 상대 경로로 삽입한다.
   ```markdown
   ![image_1](../images/{paper_name}/image_1.png)
   ```
3. **굵은 글씨**: 핵심 개념, 중요한 결론은 `**볼드**`로 강조한다.
4. **리스트**: 구조화된 정보는 불릿 리스트로 정리한다. 들여쓰기로 계층을 표현한다.
5. **수식**: LaTeX 인라인(`$...$`) 또는 블록(`$$...$$`)을 사용한다.
6. **표**: 비교 정보가 있으면 마크다운 테이블을 활용한다.
7. **인용 블록**: 핵심 인사이트나 부연 설명에 `> ...` 또는 `<aside>` 블록을 사용한다.
8. **화살표 요약**: 복잡한 개념을 한 줄로 요약할 때 `→ **핵심 요약**` 패턴을 사용한다.

#### 이미지 설명 원칙

논문 figure를 삽입할 때는 **반드시 상세한 한국어 해석**을 함께 제공한다. 이것이 이 문서의 핵심 가치다.

1. **그래프/차트**: 축의 의미, 각 선/막대의 의미, 핵심 트렌드를 설명한다.
   - 예: "가로축의 N=4, 8, 16은 참여하는 에이전트의 수입니다. 그룹의 인원수가 많아질수록 초록색 막대가 점점 낮아지는 것을 볼 수 있습니다."
2. **아키텍처 다이어그램**: 각 구성요소의 역할과 데이터 흐름을 순서대로 설명한다.
   - 예: "1. Broadcast (전파): 에이전트가 자신이 제안하는 값과 그 논리적 이유를 다른 모든 에이전트에게 널리 보냅니다."
3. **비교 실험 그래프**: 각 조건의 결과를 대비시키며 **"왜 이런 차이가 나는지"**까지 해석한다.
4. figure를 단순히 "Figure 3은 성능 비교이다"라고 쓰지 않는다. 독자가 figure를 보지 않아도 내용을 이해할 수 있을 만큼 상세하게 설명한다.

#### Introduction 작성 패턴

Introduction은 논문의 전체 스토리를 압축한 섹션이다. 아래 구조를 따른다:

```markdown
# 1. Introduction

---

### 1. 배경 및 문제 제기
- 이 분야의 현재 상황과 한계점
- 기존 접근법의 구체적인 문제

### 2. 제안: [방법 이름]
- 핵심 아이디어를 직관적으로 설명
- 기존 방법과의 차별점

### 3. 실험 결과 / 주요 기여
- 가장 인상적인 수치와 발견
- 실용적 의의
```

#### Experiments 작성 패턴

실험 결과는 **수치 나열이 아니라 해석과 인사이트 중심**으로 작성한다.

- 각 실험 설정을 먼저 간결하게 설명
- 결과 figure를 삽입하고 상세 해석
- **"왜 이런 결과가 나왔는지"**에 대한 분석 포함
- 핵심 발견에는 볼드 강조 또는 💡 마크

### Phase 5: 반복 개선 (최소 3회)

작성한 문서를 기존 예시 3개와 비교하며 품질을 높인다. 이 과정이 최종 품질을 결정한다.

**각 반복에서 점검할 항목:**

1. **구조 비교**: 기존 예시(`data/outputs/md/` 내 파일들)를 다시 읽고, 자신이 작성한 문서의 구조가 동일한 패턴인지 확인한다.
2. **설명 깊이**: 핵심 개념에 대한 설명이 기존 예시만큼 친절한지 확인한다. ML 전문가이지만 이 논문의 세부 분야는 생소한 독자를 상상하며 검토한다.
3. **이미지 품질**: 각 이미지가 해당 설명과 잘 대응하는지, 누락된 중요 figure가 없는지 확인한다.
4. **컴팩트함**: 불필요하게 장황한 부분, 핵심이 아닌 내용이 섞여있지 않은지 확인한다.
5. **가독성**: 볼드, 리스트, 표, 인용 블록 등이 적절히 활용되어 시각적으로 읽기 좋은지 확인한다.
6. **일관성**: 용어 사용, 섹션 번호 매기기, 이미지 참조 방식이 기존 예시와 일관되는지 확인한다.

**반복 개선 절차:**

```
1회차: 초안 작성 → 기존 예시와 비교 → 구조/포맷 수정
2회차: 설명 깊이와 친절함 개선 → 이미지 배치 최적화
3회차: 최종 다듬기 → 오탈자, 일관성, 컴팩트함 점검
```

매 반복마다 반드시 기존 예시 파일 중 최소 1개를 다시 읽어서 톤과 스타일을 상기한다.

### Phase 6: 로컬용 + GitHub용 두 개 파일 생성

Phase 5까지 완성된 md 파일을 기반으로, **이미지 경로만 다른 두 개의 md 파일**을 최종 생성한다.

**생성할 파일 2개:**

| 파일 | 이미지 경로 패턴 | 용도 |
|---|---|---|
| `data/outputs/md/{paper_name}_local.md` | `![image_N](../images/{paper_name}/image_N.png)` | 로컬에서 이미지 바로 확인용 |
| `data/outputs/md/{paper_name}.md` | `![image_N](https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/{paper_name}/image_N.png)` | GitHub 배포용 (최종본) |

**처리 방법:**

1. Phase 4~5에서는 **로컬 상대 경로**(`../images/{paper_name}/image_N.png`)로 md를 작성한다.
2. 완성된 md를 `data/outputs/md/{paper_name}_local.md`로 저장한다.
3. 해당 파일의 내용을 복사한 뒤, 이미지 경로를 GitHub raw URL로 치환하여 `data/outputs/md/{paper_name}.md`로 저장한다.

```bash
cp data/outputs/md/{paper_name}_local.md data/outputs/md/{paper_name}.md
sed -i "s|\.\./images/${PAPER_NAME}/|https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/${PAPER_NAME}/|g" data/outputs/md/{paper_name}.md
```

또는 Python으로:
```python
local_path = f"data/outputs/md/{paper_name}_local.md"
github_path = f"data/outputs/md/{paper_name}.md"
content = open(local_path).read()
content = content.replace(
    f"../images/{paper_name}/",
    f"https://raw.githubusercontent.com/ohilikeit/automd/master/data/outputs/images/{paper_name}/"
)
open(github_path, "w").write(content)
```

최종적으로 두 파일 모두 존재하는지 확인하고, 각각의 이미지 링크가 올바른 패턴인지 검증한다.

---

## 실행 순서 요약

사용자가 `data/pdf/{paper_name}.pdf` 파일을 지정하면:

1. **PDF 읽기**: `Read` 도구로 PDF 전체 내용 파악 (페이지 분할 필요시 나눠서)
2. **구조 설계**: 핵심 내용 기반으로 문서 뼈대 구성
3. **예시 참조**: `data/outputs/md/` 내 기존 예시 최소 1개를 읽어서 스타일 확인
4. **Figure 추출**:
   - `--mode info`로 PDF 정보 확인
   - `--mode page`로 figure가 있는 페이지 렌더링
   - vision으로 확인 후 bbox 파악
   - `--mode crop`으로 정밀 캡쳐 → `data/outputs/images/{paper_name}/`에 저장
   - **[필수]** 추출된 모든 이미지를 `Read`(vision)로 하나씩 열어서 검증. 불합격 시 재캡쳐
5. **md 작성**: 로컬 이미지 경로(`../images/{paper_name}/image_N.png`)로 문서 작성
6. **반복 개선**: 최소 3회 반복하며 예시와 비교, 품질 향상
7. **두 개 파일 생성**:
   - `data/outputs/md/{paper_name}_local.md` — 로컬 상대 경로 (작성 원본)
   - `data/outputs/md/{paper_name}.md` — GitHub raw URL 변환 (배포본)
8. **최종 확인**: 두 파일 모두 이미지 링크 패턴이 올바른지 점검

---

## 주의사항

- PDF가 매우 길 경우(30페이지 이상), 핵심 섹션(Introduction, Method, Experiments)을 우선 읽고 나머지는 필요시 참조한다.
- 논문마다 구조가 다르므로, 위 패턴을 기계적으로 적용하지 말고 **해당 논문의 핵심 기여에 맞게 유연하게** 조정한다.
- figure 추출 시 scan 모드에서 이미지가 없으면(벡터 그래픽 기반 논문), page 렌더링 후 crop 방식으로 추출한다.
- 이미지 파일명은 반드시 `image_1.png`, `image_2.png`, ... 순서를 따른다.
- 추출한 이미지는 `data/outputs/images/{paper_name}/` 디렉토리에 저장한다. 디렉토리가 없으면 자동 생성된다.

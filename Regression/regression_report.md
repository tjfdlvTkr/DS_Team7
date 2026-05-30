# 스마트폰 가격 회귀 · 앙상블 실험 요약

이 문서는 `Regression/` 작업 공간에서 수행한 **가격 회귀(Regression)** 결과를, 보고서·발표용으로 읽기 쉽게 정리한 것입니다.

---

## 한 줄로 말하면

**전처리된 스마트폰 스펙(94개 피처)으로 유로 가격(`target_price_eur`)을 예측**했고, 여러 모델을 붙인 **Stacking 앙상블**이 가장 잘 맞았습니다. 테스트 기준 평균 오차는 **약 51유로(MAE)** 수준입니다.

---

## 1. 무엇을 예측했나? (타깃)

| 항목 | 내용 |
|------|------|
| **타깃 변수** | `target_price_eur` — 스마트폰 **출시·시장 가격(유로, EUR)** |
| **학습 방식** | 기본값 **`log_price_eur`** — 가격에 `log(1+x)` 변환 후 학습, 예측은 다시 EUR로 복원 |
| **왜 로그?** | 저가·고가가 한꺼번에 섞여 있어서, 그대로 학습하면 고가 폰에 끌려가기 쉬움. 로그를 쓰면 **가격 분포를 덜 치우치게** 맞출 수 있음 |

입력은 **모델명 검색이 아니라**, RAM·저장용량·해상도·브랜드 그룹 등 **숫자·원핫으로 정리된 스펙 표**입니다.

---

## 2. 어떤 데이터를 썼나?

| 항목 | 값 |
|------|-----|
| **파일** | `content/gsm_processed_all(price_prediction).csv` |
| **전체 행 수** | 6,251대 |
| **피처 수** | 94개 (`num__*`, `cat__*` 등 전처리 결과) |
| **학습 / 테스트** | 5,000 / **1,251** (80% · 20%, `random_state=42`) |
| **제외한 컬럼** | `oem`, `model`, `value_score` 등 — **식별자·누수 가능** 파생 지표 |

---

## 3. 어떤 모델로 회귀했나?

### 3.1 단독 모델 (Base)

각 모델이 **같은 X, 같은 타깃**으로 따로 학습·평가되었습니다.

| 모델 | 종류 | 비고 |
|------|------|------|
| **Ridge** | 선형 회귀 | 빠른 베이스라인 |
| **Random Forest** | 트리 앙상블 | 비선형, CPU |
| **LightGBM** | 그래디언트 부스팅 | `modeling` 파이프라인과 동일 계열 |
| **XGBoost** | 그래디언트 부스팅 | 이번 실험에서 추가 |
| **CatBoost** | 그래디언트 부스팅 | 이번 실험에서 추가 |

### 3.2 앙상블 (최종 후보)

| 방식 | 이름 (결과 파일 기준) | 설명 |
|------|----------------------|------|
| **Stacking** | `stacking_ridge_meta` | 베이스 5개의 **5-fold CV 예측**을 모아, **Ridge**가 최종 가격을 조합 |
| **Weighted blend** | `weighted_blend_cv` | 각 베이스의 **CV MAE 역수**로 가중 평균 |

---

## 4. 평가는 어떻게 했나?

| 단계 | 내용 |
|------|------|
| **분할** | Hold-out — 학습에 쓰지 않은 **테스트 1,251대**만으로 최종 점수 산출 |
| **베이스라인** | 학습셋 가격 **중앙값**만 예측하는 단순 기준 (MASE 비교용) |
| **주요 지표** | **MAE**, RMSE, **R²**, sMAPE, MASE |
| **앙상블 내부** | Stacking·가중 평균의 가중치 산정 시 **5-fold 교차검증(CV)** 사용 |

**MAE**를 가장 직관적으로 보면 됩니다. “실제 가격과 예측 가격이 평균적으로 몇 유로 차이 나는가”에 가깝습니다.

---

## 5. 결과 (테스트셋 기준)

### 5.1 모델별 성능 (낮을수록 좋음: MAE / RMSE)

| 순위 | 모델 | 유형 | **MAE (EUR)** | **R²** | RMSE (EUR) |
|:----:|------|------|---------------|--------|------------|
| 1 | **stacking_ridge_meta** | 앙상블 | **50.83** | **0.40** | 183.42 |
| 2 | lightgbm | 단독 | 51.40 | 0.39 | 185.11 |
| 3 | xgboost | 단독 | 51.44 | 0.39 | 184.94 |
| 4 | weighted_blend_cv | 앙상블 | 51.65 | 0.39 | 185.12 |
| 5 | catboost | 단독 | 52.25 | 0.39 | 184.20 |
| 6 | random_forest | 단독 | 53.50 | 0.38 | 186.83 |
| 7 | ridge | 단독 | 63.16 | 0.33 | 194.06 |

### 5.2 보조 지표

| 모델 | sMAPE (%) | MASE |
|------|-----------|------|
| **stacking_ridge_meta** | 22.84 | 0.45 |
| lightgbm | 23.03 | 0.46 |
| ridge | 28.91 | 0.57 |

- **R² ≈ 0.40** → 가격 변동의 약 40%를 스펙만으로 설명. “완벽한 단가표”보다는 **가이드라인**에 가깝다.
- **MASE < 1** → 중앙값만 찍는 것보다 낫다.

### 5.3 가중 평균(Weighted blend) 비중

CV에서 잘 맞은 모델일수록 비중을 더 줬습니다.

| 베이스 모델 | blend 비중 |
|-------------|------------|
| **catboost** | 21.0% |
| **lightgbm** | 20.9% |
| **xgboost** | 20.8% |
| **random_forest** | 20.1% |
| **ridge** | 17.1% |

부스팅 3종이 비슷한 비중이고, 선형 Ridge는 가장 낮습니다.

---

## 6. 무엇이 최종 “타깃 모델”인가?

보고·재현 기준으로 추천하는 **최종 모델**은 아래와 같습니다.

| 구분 | 선택 |
|------|------|
| **최종 예측 모델** | **`stacking_ridge_meta`** (Stacking + Ridge 메타) |
| **저장 파일** | `Regression/models/ensemble_price_regressor.pkl` |
| **타깃 변환** | **`log_price_eur`** |

단독으로만 쓴다면 **LightGBM** 또는 **XGBoost**도 성능이 비슷하지만, 이번 hold-out에서는 **Stacking이 MAE·R² 모두 1위**였습니다.

---

## 7. 실행 환경 메모

| 항목 | 내용 |
|------|------|
| **실행 명령** | `py Regression/run_ensemble_regression.py --device gpu` |
| **GPU** | RTX 5060 Ti 요청했으나, 당시 환경에서 **CUDA 미감지 → CPU로 학습** (`run_metadata.json`) |
| **소요** | Stacking 약 23초, 전체 파이프라인 수십 초 수준 (CPU) |

---

## 8. 산출물 위치

| 파일 | 용도 |
|------|------|
| `outputs/ensemble_holdout_metrics.csv` | 모델별 지표 표 |
| `outputs/ensemble_holdout_predictions.csv` | 테스트 실제가 vs stacking / blend 예측 |
| `outputs/blend_weights.json` | 가중 평균 비중 |
| `outputs/plots/ensemble_holdout_mae.png` | MAE 비교 그래프 |
| `outputs/run_metadata.json` | 데이터 규모·디바이스·베이스 목록 |

---

## 9. 해석 시 주의할 점

1. **평균 오차 51유로**는 “대부분의 폰이 ±50유로 안쪽”이 아니라, **전체 평균**입니다. `ensemble_holdout_predictions.csv`를 보면 일부 기종은 훨씬 크게 빗나갈 수 있습니다.
2. **EUR·GSM 전처리 데이터** 기준이라, 지역·프로모션·실매가는 반영되지 않습니다.
3. **`modeling/`의 B2B 가격 가이드**는 별도 `best_price_regressor.pkl`(단독 LightGBM)을 쓰며, 이 Regression 앙상블과 **파일은 다릅니다**. 가격 회귀 실험·과제 보고용은 **`Regression/` 결과**를 기준으로 하면 됩니다.

---

## 10. 결론

| 키워드 | 요약 |
|--------|------|
| **타깃** | `target_price_eur` (로그 학습) |
| **회귀 모델** | Ridge, RF, LightGBM, XGBoost, CatBoost + **Stacking / Weighted blend** |
| **평가** | 80/20 hold-out, MAE·R² 중심 |
| **최고 성능** | **Stacking (`stacking_ridge_meta`)**, MAE **50.83 EUR**, R² **0.40** |
| **실무적 의미** | 신규 기종 스펙을 넣었을 때 **대략적인 EUR 가격대**를 제시하는 데 적합한 수준 |

재실행: `py Regression/run_ensemble_regression.py --device auto`  
상세 사용법: `Regression/guidance.md`

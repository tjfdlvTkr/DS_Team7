# DS Team 7 GSM 모델링 요약

## 사용 데이터
- 입력 위치: `content/`
- 회귀 학습: `content/gsm_processed_all(price_prediction).csv`
- 추천/클러스터링: `content/gsm_processed(recommendation).csv`
- 가격 단위: EUR
- 회귀 타겟: `target_price_eur`

## Task A. 가격 예측 회귀
최종 holdout 기준 최고 모델은 **LightGBM (log_price_eur)** 이다.
MAE는 **51.73 EUR**, R2는 **0.3927** 이다.

상위 가격 결정 요인:
num__resolution_total_px, num__storage_gb, num__resolution_width_px, num__body_weight_g, num__screen_to_body_pct

## Task B. 세그먼트 클러스터링
KMeans k=3으로 Entry / Mid-range / Flagship을 만들었다.

| segment | count | avg_price_eur | median_price_eur | avg_spec_score_0_100 | avg_ram_gb | avg_storage_gb | avg_battery_capacity_mah | avg_main_camera_max_mp | avg_ppi | avg_network_generation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entry | 1026 | 200.4800 | 180.0000 | 31.7004 | 2.5478 | 32.0939 | 3591.4324 | 11.4714 | 291.3519 | 3.8903 |
| Mid-range | 825 | 325.4785 | 300.0000 | 37.9323 | 4.2303 | 80.4000 | 3433.5857 | 17.1592 | 425.5612 | 4.0000 |
| Flagship | 262 | 589.9252 | 500.0000 | 51.6283 | 8.6107 | 275.2977 | 4445.4330 | 45.6260 | 405.5954 | 4.3969 |

## Task C. 가성비 이상치와 추천
가성비 우수 모델은 세 가지 기준을 결합해 식별한다.
1. 같은 세그먼트 내 `value_score` Z-score 이상치
2. `IsolationForest` 기반 이상치 중 value score 상위 후보
3. 하드웨어 스펙으로 예측한 `expected_market_price_eur`보다 실제 가격이 낮은 market underpricing 이상치

추천은 사용자 예산 이하 후보만 남긴 뒤 `performance_to_price_ratio`를 첫 번째 정렬 기준으로 최적 대안을 고른다.

| is_value_outlier | count | avg_price_eur | avg_value_score | avg_performance_to_price_ratio | avg_value_lift_vs_segment | avg_price_discount_vs_segment_median | avg_expected_market_price_eur | avg_market_price_gap_eur | avg_market_price_discount_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 1943 | 312.9693 | 14.8337 | 0.1483 | 0.9154 | -0.1654 | 305.7433 | -7.2260 | -0.0354 |
| True | 170 | 121.6080 | 32.8320 | 0.3283 | 1.9665 | 0.5141 | 206.7662 | 85.1581 | 0.3792 |

추천 검증:

| scenario | scenario_label | budget_eur | row_count | max_price_eur | all_within_budget | is_value_score_descending | is_performance_to_price_ratio_descending | top_value_score | top_performance_to_price_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 학생 예산 | 200 | 10 | 71.5000 | True | True | True | 59.2350 | 0.5923 |
| 2 | 일반 직장인 | 400 | 10 | 153.9890 | True | True | True | 37.2435 | 0.3724 |
| 3 | 프리미엄 유저 | 800 | 10 | 223.1907 | True | True | True | 29.1731 | 0.2917 |
| 4 | 입문자/세컨드폰 | 150 | 10 | 71.5000 | True | True | True | 59.2350 | 0.5923 |

## 직접 실행
```bash
cd modeling
python3 two_way_solution.py --mode demo
python3 two_way_solution.py --mode recommend --budget-eur 400 --top-n 5
```

## 주요 산출물
- `models/best_price_regressor.pkl`
- `outputs/regression_holdout_metrics.csv`
- `outputs/regression_cv_metrics.csv`
- `outputs/regression_cv_fold_metrics.csv`
- `outputs/feature_importance_best_target.csv`
- `outputs/brand_premium_summary.csv`
- `outputs/adjusted_brand_premium_summary.csv`
- `outputs/df_with_segments.csv`
- `outputs/value_outliers.csv`
- `outputs/value_outlier_method_comparison.csv`
- `outputs/recommendations_all_scenarios.csv`
- `outputs/modeling_quality_checks.csv`
- `outputs/two_way_business_price_guides.csv`
- `outputs/two_way_user_recommendations.csv`
- `docs/02_MODEL_EVALUATION_HANDOFF.md`

## 한계
- 가격은 EUR 기준이며 지역/통신사/유통 채널 차이는 반영하지 않는다.
- 일부 스펙은 텍스트에서 파싱된 값이므로 원본 표기 품질의 영향을 받는다.
- GSM 최종 전처리 파일은 CPU 벤치마크를 직접 제공하지 않아 RAM, storage, camera, battery, network, display 계열 피처와 `spec_score_0_100`을 하드웨어 성능 대리 지표로 사용한다.
- 구형 모델과 최신 모델이 함께 있어 `phone_age` 해석이 필요하다.

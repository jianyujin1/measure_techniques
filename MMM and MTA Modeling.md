
# MMM & MTA Foundation: Theories and Python Code

This repository provides detailed theoretical explanations and Python implementations of core concepts required for Marketing Mix Modeling (MMM) and Multi-Touch Attribution (MTA).

---

## 1. Linear & Logistic Regression

**Topic**: Linear & Logistic Regression  
**Area**: Statistics, Machine Learning  
**Why it’s Important**: These are foundational models for estimating marketing impact (MMM) and predicting user conversion (MTA).

### English Explanation:
- **Linear Regression** models the relationship between a continuous dependent variable and one or more independent variables. It fits a line that minimizes the sum of squared differences between actual and predicted values.
- **Logistic Regression** is used when the dependent variable is binary (e.g., convert vs. not convert). It estimates the probability of a class using the logistic function and is widely used in classification problems.

### 한국어 설명:
- **선형 회귀**는 연속적인 종속 변수와 하나 이상의 독립 변수 간의 관계를 모델링하며, 실제 값과 예측 값 간의 오차 제곱합을 최소화하는 직선을 찾습니다.
- **로지스틱 회귀**는 종속 변수가 0 또는 1과 같은 이진형일 때 사용되며, 로지스틱 함수를 통해 특정 클래스에 속할 확률을 예측합니다.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

# Linear Regression Example
X = np.array([[1], [2], [3], [4]])  # Independent variable
y = np.array([2, 4, 6, 8])          # Dependent variable
model = LinearRegression().fit(X, y)
print("Linear Coefficient:", model.coef_)  # Output: 2 (slope of the line)

# Logistic Regression Example
X_log = np.array([[0], [1], [2], [3]])      # Binary feature
y_log = np.array([0, 0, 1, 1])              # Labels
log_model = LogisticRegression().fit(X_log, y_log)
print("Logistic Coefficients:", log_model.coef_)  # Outputs weight for classification
```

---

## 2. Time Series Analysis

**Topic**: Time Series Analysis  
**Area**: Statistics  
**Why it’s Important**: Time-based marketing data often has trends, seasonality, and lag effects that must be modeled for MMM accuracy.

### English Explanation:
Time series analysis examines data points indexed in time order. Key elements include:
- **Trend**: Long-term increase or decrease
- **Seasonality**: Periodic fluctuations (e.g., weekend dips)
- **Lag Effect**: Delayed impact from media or promotions

### 한국어 설명:
시계열 분석은 시간 순으로 정렬된 데이터를 분석합니다. 주요 요소는 다음과 같습니다:
- **추세 (Trend)**: 장기적인 증가 또는 감소 경향
- **계절성 (Seasonality)**: 주기적인 변동 (예: 주말에 하락)
- **지연 효과 (Lag Effect)**: 마케팅 활동의 시차 효과

```python
import pandas as pd
import matplotlib.pyplot as plt

# Time series data (monthly sales)
ts = pd.Series([100, 120, 130, 150, 170], index=pd.date_range("2024-01-01", periods=5, freq='M'))
ts.plot(title="Time Series Example - Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()
plt.show()
```

---

## 3. Bayesian Statistics

**Topic**: Bayesian Statistics  
**Area**: Statistics  
**Why it’s Important**: It allows incorporating prior knowledge and estimating uncertainty in model predictions (e.g., media effectiveness in MMM).

### English Explanation:
Bayesian inference updates the probability of a hypothesis as new evidence is introduced. It combines:
- **Prior**: Belief before observing data
- **Likelihood**: Data evidence
- **Posterior**: Updated belief

### 한국어 설명:
베이지안 통계는 새로운 데이터를 관찰함에 따라 가설의 확률을 업데이트합니다. 구성 요소는 다음과 같습니다:
- **사전 확률 (Prior)**: 관측 전의 믿음
- **우도 (Likelihood)**: 관측된 데이터의 가능성
- **사후 확률 (Posterior)**: 업데이트된 확률

```python
import pymc as pm

# Bayesian inference example
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=1)  # Prior belief about alpha
    obs = pm.Normal("obs", mu=alpha, sigma=1, observed=[1.2, 0.9, 1.3])  # Observed data
    trace = pm.sample(100, chains=1)

print(trace["alpha"].mean())  # Posterior mean
```

---

## 4. Causal Inference

**Topic**: Causal Inference  
**Area**: Econometrics, ML  
**Why it’s Important**: Helps isolate the true effect of a campaign or treatment from confounding variables.

### English Explanation:
Causal inference determines whether a treatment (e.g., ad spend) causes an outcome (e.g., sales). It tries to avoid spurious correlations.

### 한국어 설명:
인과 추론은 특정 요인(예: 광고 지출)이 결과(예: 매출)에 영향을 주는지를 분석합니다. 상관관계가 아닌 인과관계를 추정하는 것이 핵심입니다.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Basic A/B test style causal estimate
data = pd.DataFrame({
    'treatment': [1, 0, 1, 0, 1],  # 1 = treated, 0 = control
    'outcome': [200, 180, 220, 170, 210]
})
X = data[['treatment']]
y = data['outcome']
model = LinearRegression().fit(X, y)
print("Causal Effect Estimate (ATE):", model.coef_)  # Difference in means
```

---

## 5. Game Theory (Shapley Value)

**Topic**: Game Theory (Shapley Value)  
**Area**: Mathematics, Optimization  
**Why it’s Important**: Assigns fair credit for conversions across multiple touchpoints in MTA.

### English Explanation:
The Shapley value assigns a fair value to each "player" (channel) by computing their average marginal contribution across all permutations.

### 한국어 설명:
샤플리 값은 각 채널이 전환에 기여한 정도를 모든 가능한 순열에서 평균 기여도로 계산하여 공정하게 분배합니다.

```python
from itertools import permutations
from collections import defaultdict

# Simple Shapley calculator
def shapley_credit(channels):
    perms = list(permutations(channels))
    shapley = defaultdict(float)
    for perm in perms:
        for i, ch in enumerate(perm):
            shapley[ch] += 1 / len(perm)
    for ch in shapley:
        shapley[ch] /= len(perms)
    return shapley

print(shapley_credit(['email', 'search', 'social']))  # Example Shapley values
```

---

## 6. Markov Chains

**Topic**: Markov Chains  
**Area**: Probability Theory  
**Why it’s Important**: Used in MTA to model user paths and conversion probabilities.

### English Explanation:
Markov chains represent systems that transition from one state to another with fixed probabilities. Used in attribution to measure removal effects of channels.

### 한국어 설명:
마르코프 체인은 고정된 전이 확률로 상태 간 이동하는 시스템을 모델링하며, 전환 경로 분석 및 기여도 측정에 사용됩니다.

```python
import numpy as np

# Example transition matrix for attribution
states = ['email', 'search', 'conversion']
P = np.array([
    [0.1, 0.6, 0.3],  # From email
    [0.0, 0.2, 0.8],  # From search
    [0.0, 0.0, 1.0]   # Conversion (absorbing state)
])

print("Probability of moving from 'search' to 'conversion':", P[1][2])
```


---

## Summary Table: Comparison of Techniques

| Technique            | Type         | Used in          | Key Feature                          | Pros                                           | Cons                                        |
|----------------------|--------------|------------------|---------------------------------------|------------------------------------------------|---------------------------------------------|
| Linear Regression    | Statistical  | MMM              | Models continuous output              | Simple, interpretable, good for trend analysis | Assumes linearity, sensitive to outliers    |
| Logistic Regression  | Statistical  | MTA              | Classifies binary outcomes            | Useful for conversion prediction               | Only models linear decision boundaries      |
| Time Series Analysis | Statistical  | MMM              | Captures seasonality & trends         | Handles temporal patterns                      | May require stationarity, sensitive to noise|
| Bayesian Statistics  | Probabilistic| MMM              | Uses priors and estimates uncertainty | Models belief update, handles uncertainty      | Requires computational power and priors     |
| Causal Inference     | Econometric  | MMM              | Measures true treatment effect        | Identifies causality, not just correlation     | Requires good experimental design            |
| Shapley Value        | Game Theory  | MTA              | Fair value distribution among players | Fair, interpretable, handles many touchpoints  | Computationally expensive for many channels |
| Markov Chains        | Probabilistic| MTA              | State transition-based path modeling  | Captures sequential behavior, removal effects  | Ignores time gaps, assumes memorylessness   |

---

These techniques together form the backbone of robust MMM and MTA modeling.  
For most real-world attribution and planning systems, a **hybrid approach** combining time series + causal + user-level attribution is ideal.

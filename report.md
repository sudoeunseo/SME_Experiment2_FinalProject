# 스마트모빌리티공학실험2 Final Project Report

| 항목 | 내용 |
|---|---|
| 이름 | 김은서 |
| 학번 | 12241079 |
| 분반 | class1 / 1분반 |
| 과목 | 스마트모빌리티공학실험2 Final Project |
| 주제 | RTT 기반 positioning/localization |
| 최종 모델명 | V12A_SafeResidualMemory |

## 1. 모티베이션 & 인트로

본 프로젝트의 목표는 18개 기지국에서 측정된 RTT 기반 거리 추정값 d_hat을 이용하여 사용자 단말의 2차원 위치를 추정하는 것이다. RTT 기반 위치추정은 각 기지국과 사용자 사이의 거리 정보를 활용하므로, 이상적인 경우에는 multilateration 또는 least-squares 기반 기하학적 위치추정으로 접근할 수 있다. 그러나 실제 실내 환경에서는 측정 거리값이 항상 실제 거리와 일치하지 않는다. 벽, 기둥, 가구, 사람 등에 의해 신호가 직접 경로가 아닌 반사·우회 경로로 도달하면 NLOS 오차가 발생하고, 이때 RTT 기반 거리는 실제 거리보다 길게 측정되는 경향이 있다. 또한 anchor별 bias, scale 차이, 일부 거리값의 NaN/inf/음수, 특정 anchor의 큰 outlier가 함께 존재할 수 있다. 따라서 본 문제는 단순히 여러 원의 교점을 찾는 문제가 아니라, 어떤 거리값을 얼마나 믿을 것인지와 어떤 위치추정 모델을 어떤 sample에서 사용할 것인지를 동시에 판단해야 하는 문제이다.

중간발표 단계에서는 WiFi RTT와 UWB hybrid 실내 위치추정 문제를 대상으로, kNN fingerprinting, WLS, Robust NLS, Random Forest/Extra Trees, fixed stacking ensemble을 비교하였다. 당시 실험의 핵심 관찰은 다음과 같았다. 첫째, kNN fingerprinting은 거리값을 물리식으로 직접 해석하지 않고, 거리 패턴이 유사한 reference point를 찾기 때문에 실내 multipath 환경에서 안정적인 경우가 있었다. 둘째, WLS와 Robust NLS는 anchor geometry를 활용할 수 있지만, 신뢰도가 낮은 anchor의 거리값을 그대로 사용하면 예측 위치가 outlier 방향으로 끌릴 수 있었다. 셋째, median, MAD, IQR, residual, variance, mean-median difference 같은 통계량은 NLOS 또는 측정 불안정성을 직접 label 없이 추정하는 데 유용했다. 넷째, OOF 방식은 각 sample을 예측할 때 해당 sample의 정답 좌표를 calibration이나 모델 학습에 직접 사용하지 않도록 하므로, 단순 in-sample 평가보다 data leakage 위험을 줄이는 데 필요했다.

중간발표 당시 최종 구조는 여러 base model의 예측 좌표를 고정 가중치로 결합하는 fixed stacking ensemble에 가까웠다. 당시에는 Weighted kNN, Simple Ensemble, Reliability Robust NLS, Final Stacking OOF를 비교했고, 여러 모델을 결합했을 때 단일 모델보다 안정적인 성능을 얻을 수 있다는 점을 확인했다. 그러나 교수님 피드백을 통해 fixed stacking 방식의 한계도 명확해졌다. 예를 들어 kNN의 k를 어떻게 정했는지, WLS의 anchor별 weight를 어떻게 설정했는지, weight grid를 충분히 탐색했는지, Robust NLS에서 큰 오차와 작은 오차를 binary threshold로 나눈다면 그 경계값 자체가 불안정한 hyperparameter가 아닌지, 여러 모델의 예측을 단순 고정 비율로 평균하는 것이 실제로 sample별 상황을 반영하는지에 대한 지적이 있었다. 특히 '모델 여러 개를 단순 averaging하지 말고, 상황에 따라서 모델을 골라서 써야 한다'는 피드백은 최종 알고리즘 방향을 바꾸는 중요한 계기가 되었다.

이 피드백을 바탕으로 최종 프로젝트에서는 고정 가중치 기반 ensemble을 그대로 확장하지 않았다. 대신 각 sample에서 어떤 expert가 더 신뢰할 만한지를 OOF 결과로 학습하는 방향으로 설계를 변경했다. 같은 RTT 입력이라도 어떤 sample은 fingerprint 기반 tree regressor가 안정적이고, 어떤 sample은 calibration된 거리와 geometry residual이 잘 맞는 NLS 계열 expert가 유리할 수 있다. 또한 어떤 sample은 expert들이 서로 비슷한 위치를 예측하여 단순 평균이 큰 문제가 없지만, 어떤 sample은 expert disagreement가 커져서 실패한 expert의 영향을 줄이는 것이 중요하다. 따라서 본 프로젝트에서는 하나의 최적 모델이나 전체 sample에 동일하게 적용되는 고정 ensemble weight보다, sample별 expected error를 예측하고 그에 따라 expert contribution을 다르게 주는 구조가 더 적합하다고 판단했다.

최종 알고리즘 V12A_SafeResidualMemory는 이러한 문제의식에서 설계되었다. 전체 구조는 OOF Error-Gated Mixture-of-Experts, HGB direct meta learner, Safe Residual Memory의 3단계로 구성된다. 먼저 fingerprint 기반 expert, geometry 기반 expert, hybrid expert를 병렬로 만들고, 각 expert의 OOF prediction과 실제 localization error를 계산한다. 그다음 gate model은 거리 통계, expert disagreement, residual consistency, geometry feature를 입력으로 받아 sample별 expert expected error를 예측한다. hidden test에서는 정답 좌표를 알 수 없으므로 실제 error가 아니라 gate model이 예측한 expected error를 사용하고, expected error가 작을수록 해당 expert의 softmax weight가 커지도록 하여 sample-adaptive ensemble을 만든다. 이는 중간발표의 fixed stacking에서 한 단계 발전한 구조로, 모든 sample에 같은 weight를 쓰는 것이 아니라 sample마다 expert weight가 달라지는 점이 핵심이다.

또한 최종 모델은 단순히 expert prediction을 weighted average하는 데서 끝나지 않는다. Gated MoE 이후에도 expert prediction, gate weight, predicted error, residual consistency 사이에는 추가적인 보정 정보가 남아 있을 수 있으므로, OOF expert prediction과 gate feature를 결합한 meta feature를 만들고 HGB direct meta learner를 적용했다. 이 meta layer에서는 alpha 후보에 0을 포함하여, meta correction이 OOF 기준으로 base gated MoE보다 나빠지는 경우에는 보정을 약하게 하거나 사용하지 않을 수 있도록 했다. 마지막으로 Safe Residual Memory는 OOF prediction 이후에도 남는 residual pattern을 scaled meta-feature space에 저장하고, hidden sample과 유사한 OOF sample들의 residual 방향을 inverse-distance weighted 방식으로 보수적으로 보간한다. 이때 self-exclusion/LOO-style 검증으로 k와 alpha를 선택하여, 단순한 위치 lookup이나 train sample 암기가 되지 않도록 했다.

본 프로젝트의 설계 의도는 성능을 과장하거나 복잡한 모델을 무조건 많이 쌓는 것이 아니다. 중간발표와 교수님 피드백을 통해 확인한 핵심 문제는, RTT 실내 측위에서 sample별로 실패하는 모델과 신뢰할 수 있는 거리값이 달라진다는 점이었다. 따라서 최종 모델은 anchor별 range calibration으로 systematic bias를 줄이고, robust feature로 거리 패턴을 안정화하며, fingerprint expert와 geometry expert를 함께 사용하고, OOF expected-error gating으로 sample별 expert 선택 문제를 완화하도록 설계했다. 즉, V12A_SafeResidualMemory는 기존 fixed stacking의 한계였던 고정 weight averaging을 'OOF 기반 expected-error gated MoE + conservative meta correction + safe residual memory'로 확장한 모델이다.

본 보고서의 모든 성능 수치는 제공된 labeled training data에 대한 OOF 기준 결과이다. OOF는 각 sample이 자기 자신을 학습에 직접 포함하지 않은 fold model로부터 예측을 받도록 구성했기 때문에 단순 train fit 결과보다 공정하지만, 완전한 hidden test 성능을 보장하지는 않는다. Hidden test에서는 anchor 측정 분포, NLOS 비율, outlier pattern, 실행 환경이 달라질 수 있으므로, 본 보고서에서는 OOF 성능과 hidden test 예상 성능을 분리해서 해석한다. 최종 모델은 OOF 기준으로 단일 HistGBR expert와 fixed averaging보다 개선되었지만, 구조가 복잡하고 residual memory를 포함하므로 overfitting 위험도 함께 가진다. 따라서 본 프로젝트의 최종 목표는 최고 성능을 단정하는 것이 아니라, 중간발표에서 드러난 fixed stacking과 hand-tuned weight의 한계를 OOF 기반 sample-adaptive gating 구조로 개선하고, RTT localization에서 발생하는 bias, NLOS, outlier, expert disagreement를 여러 단계에서 완화하는 데 있다.

## 2. 알고리즘 설명

### 2.1 입력 표기, 출력 규격, OOF 평가 구조

본 프로젝트에서 anchor 수를 \(M=18\), 사용자 수를 \(N\)이라고 두었다. anchor \(i\)의 2차원 좌표를 \(\mathbf{b}_i=[b_{ix}, b_{iy}]^T\), 사용자 \(n\)의 실제 좌표를 \(\mathbf{p}_n=[x_n,y_n]^T\), anchor \(i\)에서 사용자 \(n\)에 대해 측정된 RTT 기반 거리값을 \(\tilde d_{i,n}\)이라고 표기한다. 최종 제출 함수는 \(N\)을 고정값으로 두지 않고 입력 \(d\_hat\)의 shape에서 읽으며, 최종 예측 행렬은 첫 행이 \(x\), 둘째 행이 \(y\)인 \((2,N)\) 형태로 반환한다.

OOF 평가는 전체 labeled training sample을 5개 fold로 나누어 수행했다. fold \(f\)의 validation index 집합을 \(V_f\), 나머지 train index 집합을 \(T_f\)라고 하면, \(n \in V_f\)인 sample을 예측할 때 calibration parameter, expert model, gate model 학습에는 \(T_f\)만 사용한다. 이 방식의 목적은 각 sample이 자기 자신의 정답 좌표를 직접 학습에 포함한 모델로 예측되는 것을 막는 것이다. 따라서 본 보고서의 OOF 성능은 단순 train-fit 성능이 아니라, fold 단위로 sample-level leakage를 줄인 내부 검증 성능이다. 다만 이는 hidden test 성능을 보장하는 것은 아니므로, 결과 해석에서는 OOF와 hidden test를 구분했다.

### 2.2 거리 측정 모델과 anchor별 calibration

TOA/RTT 기반 위치추정은 base station과 mobile station 사이의 거리 측정으로부터 위치를 추정하는 문제이다. Cheung et al.의 TOA 기반 mobile location formulation에서는 range measurement가 실제 거리 성분과 range error가 더해진 형태로 모델링된다 [9]. 본 프로젝트도 같은 관점에서 RTT 거리값을 다음과 같이 해석했다.

\[
\tilde d_{i,n} = \|\mathbf{p}_n-\mathbf{b}_i\|_2 + \beta_i + \epsilon_{i,n}
\]

여기서 \(\beta_i\)는 anchor \(i\)의 systematic bias이고, \(\epsilon_{i,n}\)은 sample별 측정 noise, multipath, NLOS, outlier를 포함한 잔차이다. 실제 코드에서는 hidden test의 \(\mathbf{p}_n\)을 알 수 없으므로, \(\beta_i\)와 scale은 labeled training fold에서만 추정한다.

학습 fold에서 anchor \(i\)와 사용자 \(n\)의 true range는 다음과 같이 계산했다.

\[
r_{i,n}^{true}=\|\mathbf{p}_n-\mathbf{b}_i\|_2
\]

anchor별 측정 오차는 다음과 같다.

\[
e_{i,n}^{range}=\tilde d_{i,n}-r_{i,n}^{true}
\]

anchor별 median bias는 평균이 아니라 median으로 추정했다.

\[
\hat \beta_i = \operatorname{median}_{n \in T_f}\left(e_{i,n}^{range}\right)
\]

MAD 기반 robust scale은 다음과 같이 계산했다.

\[
\hat \sigma_i = 1.4826 \cdot \operatorname{median}_{n \in T_f}
\left| (e_{i,n}^{range}-\hat \beta_i) -
\operatorname{median}_{m \in T_f}(e_{i,m}^{range}-\hat \beta_i) \right|
\]

1.4826은 정규분포 가정에서 MAD를 표준편차 scale에 맞추기 위해 사용하는 상수이다. 이 scale은 특정 outlier 하나가 anchor reliability 전체를 흔드는 것을 줄이기 위한 값이다.

보정된 거리는 다음과 같이 정의했다.

\[
d^c_{i,n}=\operatorname{clip}\left(\tilde d_{i,n}-\hat\beta_i,\; q^{lo}_i,\; q^{hi}_i\right)
\]

여기서 \(q^{lo}_i\), \(q^{hi}_i\)는 train fold에서 얻은 anchor별 분위수 기반 하한과 상한이다. 즉, hidden test에서는 정답 좌표를 사용하지 않고, train fold에서 학습된 \(\hat\beta_i\), \(\hat\sigma_i\), \(q^{lo}_i\), \(q^{hi}_i\)만 적용한다.

anchor별 기본 weight는 robust scale에 반비례하도록 두었다.

\[
w_i^{base}=\frac{1}{\hat\sigma_i^2+\varepsilon}
\]

이후 너무 작은 weight나 너무 큰 weight가 생기지 않도록 train fold의 분위수 범위 안으로 clipping하고, median이 1에 가까워지도록 normalize했다. 이 weight는 교수님 피드백에서 지적된 앵커마다 weight를 어떻게 두는가에 대한 최종 답변이다. 본 최종 코드에서는 18개 anchor weight 조합을 전수조사하지 않았고, OOF train fold에서 추정한 robust scale을 이용해 anchor reliability를 연속값으로 계산했다. 따라서 weight는 binary decision이나 manual grid 조합이 아니라, anchor별 측정 안정성에서 유도된 연속 신뢰도이다.

### 2.3 Robust feature extraction

거리값만 그대로 사용하면 anchor bias와 NLOS outlier에 취약하므로, 본 모델은 보정 거리와 상대 패턴을 함께 feature로 사용했다. 사용자 \(n\)에 대한 calibrated distance vector를 \(\mathbf{d}^c_n=[d^c_{1,n},\dots,d^c_{M,n}]^T\)라고 하면, robust normalized distance는 다음과 같다.

\[
z_{i,n}=\frac{d^c_{i,n}}{\hat\sigma_i+\varepsilon}
\]

anchor 간 pairwise difference feature는 다음과 같다.

\[
\Delta z_{i,j,n}=z_{i,n}-z_{j,n}, \quad 1 \le i < j \le M
\]

이 feature는 전체 거리 scale이 조금 달라져도 anchor 사이의 상대적 거리 구조를 보존하기 위한 것이다. 또한 각 sample에 대해 anchor별 가까운 순위를 rank feature로 만들고, sorted distance, min, max, mean, median, percentile, standard deviation 등 거리 통계량을 함께 사용했다.

기하학적 prior 역할을 하는 inverse-distance weighted centroid는 다음과 같이 계산했다.

\[
\mathbf{c}^{(p)}_n =
\frac{\sum_{i=1}^{M} \mathbf{b}_i \cdot \left(\max(d^c_{i,n},0.2)\right)^{-p}}
{\sum_{i=1}^{M} \left(\max(d^c_{i,n},0.2)\right)^{-p}+\varepsilon},
\quad p \in \{1,2\}
\]

이 centroid는 최종 위치로 직접 쓰기 위한 값이 아니라, fingerprint expert와 tree expert가 사용할 수 있는 coarse geometry feature이다.

### 2.4 Fingerprint 기반 expert

Fingerprint expert는 거리 패턴이 유사한 training sample의 위치를 이용한다. Robust feature vector를 \(\mathbf{x}_n\), training sample \(m\)의 feature를 \(\mathbf{x}_m\), 좌표를 \(\mathbf{p}_m\)이라고 하면 feature distance는 다음과 같다.

\[
\delta_{n,m} = \|\mathbf{x}_n-\mathbf{x}_m\|_2
\]

KNN expert는 가장 가까운 \(k\)개의 training sample 집합을 \(\mathcal{N}_k(n)\)으로 두고, inverse-distance weight를 계산한다.

\[
a_{n,m}=
\frac{(\delta_{n,m}+10^{-6})^{-p}}
{\sum_{\ell \in \mathcal{N}_k(n)}(\delta_{n,\ell}+10^{-6})^{-p}},
\quad m \in \mathcal{N}_k(n)
\]

\[
\hat{\mathbf{p}}^{KNN}_n = \sum_{m \in \mathcal{N}_k(n)} a_{n,m}\mathbf{p}_m
\]

최종 코드에서는 KNN_rankdiff_k7_p2와 KNN_rankdiff_k21_p2를 사용했다. 즉 \(k=7\)과 \(k=21\), \(p=2\)인 두 fingerprint expert를 별도로 두었다. Local_Ridge_KNN40은 \(k=40\)개의 가까운 이웃을 선택한 뒤, 해당 local neighborhood에서 Ridge regression을 fit하여 \(\mathbf{x}_n\) 주변의 국소적인 선형 좌표 관계를 보정한다. Cover and Hart의 nearest-neighbor 관점은 가까운 feature sample이 label 추정에 정보를 준다는 아이디어를 제공하지만 [8], 본 프로젝트에서는 classification이 아니라 2D 위치 회귀와 residual memory에 확장했다.

Tree 기반 expert로는 ExtraTreesRegressor와 HistGradientBoostingRegressor를 사용했다. 이 expert들은 위 robust feature vector와 좌표 사이의 nonlinear mapping을 학습한다. ExtraTrees는 randomization을 강하게 넣은 tree ensemble 관점 [4], gradient boosting은 순차적으로 residual을 줄이는 function approximation 관점 [5]을 참고했다. 다만 본 프로젝트에서는 일반 tabular regression에 그대로 적용한 것이 아니라, RTT calibration, rank/difference feature, distance statistics, weighted centroid feature를 직접 설계하여 입력으로 사용했다.

### 2.5 Geometry 기반 robust NLS expert

Geometry expert는 보정 거리 \(d^c_{i,n}\)와 anchor 좌표 \(\mathbf{b}_i\) 사이의 물리적 관계를 이용한다. 예측 위치를 \(\mathbf{u}\)라고 할 때 anchor \(i\)의 range residual은 다음과 같다.

\[
r_i(\mathbf{u})=\|\mathbf{u}-\mathbf{b}_i\|_2-d^c_{i,n}
\]

일반 weighted least squares는 다음 목적함수를 최소화한다.

\[
\min_{\mathbf{u}} \sum_{i=1}^{M} w_i r_i(\mathbf{u})^2
\]

그러나 RTT에는 NLOS와 outlier가 존재할 수 있으므로, 본 코드에서는 일반 squared loss만 사용하지 않고 SciPy least_squares의 soft_l1 robust loss를 사용했다. SciPy 문서에서 soft_l1은 다음과 같이 정의된다 [12].

\[
\rho(z)=2(\sqrt{1+z}-1)
\]

또한 \(C=f\_scale\)일 때 robust loss는 \(C^2 \rho(f^2/C^2)\) 형태로 scaling된다 [12]. 따라서 본 코드의 robust NLS는 개념적으로 다음 목적함수를 푼다.

\[
\min_{\mathbf{u}}
\sum_{i=1}^{M}
C^2 \rho\left(
\frac{(\sqrt{w_i}r_i(\mathbf{u}))^2}{C^2}
\right)
+
\lambda_{prior}\left\|\frac{\mathbf{u}-\mathbf{u}_0}{s_{prior}}\right\|_2^2
\]

여기서 \(\mathbf{u}_0\)는 KNN, Local Ridge, ExtraTrees prediction을 섞어 만든 fingerprint prior이다. 실제 코드에서는 prior term을 별도의 residual component로 붙인다. 즉, physics solver를 완전히 독립적으로 풀지 않고 fingerprint prior 주변에서 robust NLS를 수행하여, geometry solution이 outlier anchor에 끌려 멀리 튀는 것을 줄였다.

큰 오차와 작은 오차는 hard binary threshold로 나누지 않았다. 본 코드의 residual downweighting은 다음과 같은 연속 함수이다.

\[
w_i^{(1)} =
\frac{w_i^{base}}
{1+\left(\frac{|r_i|}{\tau}\right)^2}
\]

NLOS에서는 측정 거리가 실제보다 길어지는 positive bias가 흔하므로, 예측 위치 \(\mathbf{u}\)에서의 positive residual 성분도 연속적으로 줄였다.

\[
p_i^{+}=\max\left(d^c_{i,n}-\|\mathbf{u}-\mathbf{b}_i\|_2,0\right)
\]

\[
w_i^{(2)} =
\frac{w_i^{(1)}}
{1+\left(\frac{p_i^{+}}{0.75\tau}\right)^2}
\]

Trimmed residual NLS에서는 residual의 70% 분위수보다 큰 초과분을 다시 연속 penalty로 반영한다.

\[
e_i^{trim}=\max\left(|r_i|-Q_{70}(|r|),0\right)
\]

\[
w_i^{trim} =
\frac{w_i^{(2)}}{1+\left(\frac{e_i^{trim}}{\tau}\right)^2}
\]

따라서 최종 Robust NLS는 '오차가 크다/작다'를 0 또는 1로 자르는 binary classifier가 아니라, residual 크기가 커질수록 영향력이 부드럽게 감소하는 continuous weighting 구조이다. \(\tau\)와 \(C\)는 train fold의 MAD 기반 scale에서 정해지고, OOF 결과로 전체 구조가 검증된다.

### 2.6 RANSAC-like triplet multilateration expert

RANSAC은 outlier가 있는 상황에서 일부 subset으로 후보 모델을 만들고 residual consistency로 평가하는 관점이다 [6]. 본 프로젝트에서는 완전한 무작위 sampling 대신 anchor triplet을 deterministic하게 생성하고, 각 triplet으로 후보 위치를 계산한 뒤 전체 anchor residual로 ranking했다.

세 anchor \(a,j,k\)에 대해 range equation을 서로 빼면 다음과 같은 2차항 제거 선형식이 된다.

$$
2(\mathbf{b}_a-\mathbf{b}_j)^T\mathbf{u}
=
d_j^2-d_a^2-\|\mathbf{b}_j\|_2^2+\|\mathbf{b}_a\|_2^2
$$

$$
2(\mathbf{b}_a-\mathbf{b}_k)^T\mathbf{u}
=
d_k^2-d_a^2-\|\mathbf{b}_k\|_2^2+\|\mathbf{b}_a\|_2^2
$$

이 2×2 선형식을 풀어 triplet candidate \(\mathbf{u}_{tri}\)를 만든다. 이후 모든 anchor에 대해 residual을 계산하고, 다음과 같은 robust score를 이용해 후보를 정렬했다.

$$
S(\mathbf{u}) =
\operatorname{median}(|r|)
+0.25\cdot IQR(|r|)
+0.15\cdot \operatorname{weighted\_mean}(\min(|r|,4\tau))
+0.15\cdot \operatorname{median}(p^+)
$$

상위 후보들은 score의 역수 제곱에 비례하는 weight로 평균하여 초기값을 만들고, 마지막에는 soft_l1 robust NLS로 다시 보정한다. 즉 본 프로젝트의 RANSAC expert는 random consensus를 그대로 복사한 것이 아니라, anchor geometry와 robust residual score를 결합한 deterministic RANSAC-like multilateration expert이다.

### 2.7 Dynamic expected-error gated Mixture-of-Experts

각 expert \(j\)가 sample \(n\)에 대해 예측한 좌표를 \(\hat{\mathbf{p}}_{j,n}\)이라고 하자. OOF 단계에서는 정답 \(\mathbf{p}_n\)을 알고 있으므로 expert별 OOF localization error를 다음과 같이 계산한다.

\[
e_{j,n}^{OOF}=\|\hat{\mathbf{p}}_{j,n}^{OOF}-\mathbf{p}_n\|_2
\]

Gate model은 hidden test에서도 계산 가능한 feature \(\boldsymbol{\phi}_n\)만 입력으로 받아 expert별 expected error를 예측한다.

\[
\hat e_{j,n}=g_j(\boldsymbol{\phi}_n)
\]

여기서 \(\boldsymbol{\phi}_n\)에는 거리 통계, expert disagreement, 각 expert 예측 위치의 residual statistics, geometry feature, boundary feature가 포함된다. Hidden test에서는 \(\mathbf{p}_n\)을 모르므로 \(e_{j,n}^{OOF}\)를 사용할 수 없고, gate model의 \(\hat e_{j,n}\)만 사용한다.

Mixture-of-Experts의 기본 아이디어는 여러 expert output을 gate weight로 결합하는 것이다 [1]. 본 프로젝트의 차이는 gate network가 바로 expert weight를 출력하는 것이 아니라, OOF에서 학습한 expected error를 먼저 예측하고 그 error를 softmax에 넣는다는 점이다.

\[
\alpha_{j,n}=
\frac{\exp(-\hat e_{j,n}/T)}
{\sum_{\ell=1}^{J}\exp(-\hat e_{\ell,n}/T)}
\]

\[
\hat{\mathbf{p}}^{gate}_n=
\sum_{j=1}^{J}\alpha_{j,n}\hat{\mathbf{p}}_{j,n}
\]

여기서 \(J=13\)이고, temperature \(T\)는 OOF gating-CV grid에서 선택했다. 최종 선택값은 \(T=0.5\)이다. 이 구조는 fixed averaging과 다르다. Fixed averaging은 모든 sample에 같은 expert weight를 적용하지만, 본 모델은 sample별 predicted expected error에 따라 \(\alpha_{j,n}\)이 달라진다.

### 2.8 HGB direct meta learner

Gated MoE 이후에도 expert prediction과 gate weight 사이에는 추가적인 보정 정보가 남을 수 있다. 이를 위해 각 sample \(n\)에 대해 meta feature vector \(\mathbf{s}_n\)을 만들었다. \(\mathbf{s}_n\)에는 expert 예측 좌표 flatten vector, expert 예측 평균/중앙값/표준편차, disagreement statistics, gate weight, gated prediction, predicted error, gate feature가 포함된다.

HGB direct meta learner를 \(h(\mathbf{s}_n)\)이라고 하면, direct mode에서는 meta model이 좌표를 직접 예측한다.

\[
\hat{\mathbf{p}}^{meta\_raw}_n=h(\mathbf{s}_n)
\]

최종 meta output은 base gated prediction과 direct meta prediction을 alpha로 섞는다.

\[
\hat{\mathbf{p}}^{meta}_n=
(1-\lambda)\hat{\mathbf{p}}^{gate}_n
+
\lambda \hat{\mathbf{p}}^{meta\_raw}_n
\]

최종 선택된 meta learner는 hgb_direct이고, \(\lambda=0.5\)이다. alpha grid에는 0도 포함했다. 따라서 meta learner가 OOF 기준으로 base gated MoE보다 나빠지는 경우에는 보정을 사용하지 않거나 약하게 적용할 수 있다. 이는 stacking/super learner에서 OOF prediction을 이용해 상위 모델을 학습하는 관점 [2], [3]을 따르되, 본 프로젝트에서는 단순 stacking 대신 expected-error gating 이후의 conservative meta correction으로 변형한 것이다.

### 2.9 V12A Safe Residual Memory

Meta correction 이후에도 OOF residual이 남을 수 있다. sample \(m\)의 OOF meta prediction residual을 다음과 같이 둔다.

\[
\mathbf{q}_m = \mathbf{p}_m-\hat{\mathbf{p}}^{meta,OOF}_m
\]

Safe Residual Memory는 scaled meta-feature space에서 hidden sample \(n\)과 유사한 OOF sample들을 찾고, 그 sample들의 residual direction을 보수적으로 보간한다. scaled meta feature를 \(\tilde{\mathbf{s}}_n\)이라고 하면, memory distance는 다음과 같다.

\[
D_{n,m}=\|\tilde{\mathbf{s}}_n-\tilde{\mathbf{s}}_m\|_2
\]

가까운 \(k\)개 memory sample 집합을 \(\mathcal{M}_k(n)\)이라고 할 때 residual interpolation weight는 다음과 같다.

\[
\gamma_{n,m}=
\frac{(D_{n,m}+10^{-6})^{-2}}
{\sum_{\ell \in \mathcal{M}_k(n)}(D_{n,\ell}+10^{-6})^{-2}},
\quad m \in \mathcal{M}_k(n)
\]

Residual correction은 다음과 같다.

\[
\Delta \mathbf{p}^{mem}_n =
\sum_{m \in \mathcal{M}_k(n)} \gamma_{n,m}\mathbf{q}_m
\]

최종 prediction은 다음과 같다.

\[
\hat{\mathbf{p}}^{final}_n =
\hat{\mathbf{p}}^{meta}_n + \eta \Delta \mathbf{p}^{mem}_n
\]

최종 선택값은 \(k=15\), \(\eta=0.5\)이다. OOF 검증에서는 자기 자신의 residual을 그대로 가져오는 leakage를 막기 위해 diagonal distance를 무한대로 두는 self-exclusion/LOO-style selection을 적용했다. 따라서 Safe Residual Memory는 training 위치를 직접 lookup하는 방식이 아니라, meta-feature가 유사한 sample에서 반복적으로 남는 residual 방향을 보수적으로 보정하는 layer이다.

### 2.10 추론 속도와 chunk 설정

최종 코드에서는 KNN 거리 계산의 chunk를 128로 설정했다. 이는 위치추정 수식 자체를 바꾸는 hyperparameter가 아니라, 한 번에 계산하는 test sample 수를 제한하여 메모리 피크를 줄이는 실행 안정성 설정이다. KNN 거리 계산은 개념적으로 모든 test feature와 train feature 사이의 distance matrix를 계산하므로, chunk가 너무 크면 중간 배열의 메모리 사용량이 커질 수 있다.

본 모델은 Local_Ridge_KNN40, NLS, RANSAC triplet, residual reranking, SafeAffine NLS 등 heavy expert를 포함하므로 추론 속도 측면에서 단순 모델보다 불리하다. 하지만 최종 로컬 테스트에서 700명 입력 기준 실행 시간이 10분 제한보다 낮았으므로, hidden 300명 평가에서도 제한 시간 내 실행 가능성이 높다고 판단했다. 단, hidden test의 실제 실행 시간은 채점 환경의 CPU, 메모리, Python/scikit-learn 버전에 따라 달라질 수 있으므로 결과 섹션에서는 정확도와 실행시간 trade-off를 함께 논의했다.

### 2.11 Reference 기반 설계 차이 요약

아래 표는 본 프로젝트에서 참고한 기존 연구가 원래 제안한 내용과, 본 프로젝트에서 실제로 다르게 설계·구현한 부분을 정리한 것이다. 본 프로젝트는 특정 논문 하나를 그대로 재현한 것이 아니라, RTT localization 문제의 데이터 구조에 맞게 fingerprint, robust geometry, OOF stacking, expected-error gating, residual memory를 결합했다.

| Reference | 기존 연구가 제안한 부분 | 본 프로젝트에서 참고한 부분 | 본 프로젝트에서 다르게 설계·구현한 부분 |
|---|---|---|---|
| [1] Jacobs et al. (1991) | 여러 local expert와 gating network를 함께 학습하여 입력에 따라 expert contribution을 다르게 주는 adaptive mixture-of-experts 구조를 제안했다. | sample마다 expert의 기여도가 달라져야 한다는 mixture-of-experts 관점을 참고했다. | neural gating network가 expert weight를 직접 출력하는 방식이 아니라, OOF에서 expert별 localization error를 학습하고 hidden sample에서는 predicted expected error를 softmax에 넣어 expert weight를 계산하는 error-gated MoE로 변형했다. |
| [2] Wolpert (1992) | 여러 base learner의 prediction을 meta-level 입력으로 사용하는 stacked generalization을 제안했다. | base expert prediction을 다시 상위 model의 입력으로 사용하는 stacking 관점을 참고했다. | 단순히 expert prediction을 meta model에 넣어 좌표를 바로 예측하는 구조가 아니라, OOF expert error 기반 gating을 먼저 수행하고, 그 이후 HGB direct meta correction과 residual memory를 순차적으로 적용했다. |
| [3] van der Laan et al. (2007) | cross-validation을 이용해 여러 candidate learner를 결합하는 Super Learner framework를 제안했다. | OOF/CV 기반으로 learner 조합을 평가하고 과적합 위험을 줄이는 관점을 참고했다. | convex combination만 찾는 Super Learner가 아니라, sample별 expected error를 예측해 weight를 바꾸고, meta correction과 residual memory를 추가한 구조로 설계했다. |
| [4] Geurts et al. (2006) | Extremely Randomized Trees를 통해 tree split과 threshold를 더 random하게 선택하여 ensemble 다양성을 높이는 방법을 제안했다. | ExtraTreesRegressor를 robust feature 기반 expert 및 일부 보정 model에 사용하는 근거로 참고했다. | 일반 tabular regression에 그대로 적용하지 않고, calibrated distance, rank, pairwise difference, distance statistics, weighted centroid 같은 RTT-specific feature를 직접 만들어 입력으로 사용했다. |
| [5] Friedman (2001) | Gradient boosting을 함수공간에서 residual을 순차적으로 줄이는 greedy function approximation으로 제안했다. | HistGradientBoostingRegressor 기반 nonlinear coordinate regression과 HGB direct meta learner의 근거로 참고했다. | 단순 좌표 회귀 model 하나로 사용한 것이 아니라, robust RTT feature expert와 OOF meta feature 위에서 위치 예측 및 meta correction 용도로 사용했다. |
| [6] Fischler and Bolles (1981) | RANSAC은 outlier가 있는 데이터에서 최소 subset으로 model candidate를 만들고, 많은 inlier와 합의하는 model을 선택하는 framework를 제안했다. | outlier가 섞인 anchor 거리 측정에서 subset 후보를 만들고 전체 residual consistency로 평가하는 관점을 참고했다. | 무작위 sampling 대신 anchor triplet 후보를 deterministic하게 생성하고, median residual, IQR, weighted residual, positive residual을 포함한 robust score로 ranking한 뒤 soft-L1 NLS로 보정하는 RANSAC-like multilateration expert로 구현했다. |
| [7] Huber (1964) | 큰 residual의 영향을 줄이는 robust estimation 관점을 제안했다. | outlier에 덜 민감한 loss, robust scale, residual downweighting의 이론적 배경으로 참고했다. | Huber의 location parameter 추정 문제를 그대로 푼 것이 아니라, anchor별 MAD scale, safe affine calibration, soft-L1 NLS, continuous residual downweighting으로 RTT NLOS/outlier 영향을 줄이는 데 응용했다. |
| [8] Cover and Hart (1967) | nearest-neighbor rule을 통해 feature space에서 가까운 sample이 label 추정에 정보를 준다는 관점을 제시했다. | fingerprint feature space에서 가까운 training sample을 이용하는 kNN 기반 위치 추정 관점을 참고했다. | classification 문제가 아니라 2D position regression에 사용했고, rank/difference robust feature 기반 inverse-distance weighted KNN, Local Ridge KNN, Safe Residual Memory의 residual interpolation에 확장했다. |
| [9] Cheung et al. (2004) | 여러 base station의 TOA/range measurement를 이용해 mobile location을 least-squares 방식으로 추정하는 문제를 다루었다. | range measurement를 실제 거리와 measurement error가 결합된 값으로 보고 localization residual을 최소화하는 문제 설정을 참고했다. | TOA LS/CWLS estimator를 그대로 구현하지 않고, RTT 측정값의 anchor bias, NLOS, outlier를 고려해 range calibration, fingerprint prior, robust NLS, ML-based gating을 결합했다. |
| [10] Pedregosa et al. (2011) | scikit-learn의 machine learning model, preprocessing, cross-validation 구현 체계를 제시했다. | ExtraTrees, HistGradientBoosting, preprocessing, model validation 구현 도구의 reference로 참고했다. | scikit-learn 기본 model을 단순 적용한 것이 아니라, OOF expert generation, expected-error gating, HGB direct meta learner, Safe Residual Memory를 직접 구성했다. |
| [11] Bahl and Padmanabhan (2000) | RADAR는 실내 무선 신호를 위치별 fingerprint로 저장하고, 새로운 신호와 radio map을 비교해 위치를 추정하는 방법을 제안했다. | 무선 측정값을 위치별 fingerprint로 해석하는 indoor localization 관점을 참고했다. | RADAR의 RSS radio map을 사용하지 않고, 18개 anchor의 RTT d_hat을 distance fingerprint로 해석했으며, 여기에 range calibration, geometry residual, OOF expected-error gating을 결합했다. |
| [12] SciPy Developers | least_squares에서 robust loss 중 soft_l1을 rho(z)=2*((1+z)**0.5-1)로 정의하고, f_scale로 residual scale을 조정하는 방식을 제공한다. | Robust NLS에서 큰 residual의 영향을 줄이는 soft-L1 loss 수식과 scaling 방식을 참고했다. | SciPy solver를 그대로 호출하는 것에 그치지 않고, anchor별 robust weight, fingerprint prior, positive residual downweighting, trimmed residual penalty를 함께 넣어 RTT localization용 MAP-style robust NLS expert로 사용했다. |

## 3. Agent AI 활용 방안

본 프로젝트에서는 ChatGPT를 주로 사용했고, 코드 점검 보조 용도로 Claude Code도 일부 활용했다. 다만 Agent AI가 최종 알고리즘을 대신 설계한 것은 아니며, 본인이 문제 상황, 중간발표 피드백, OOF 결과, reference 후보, 최종 코드 구조를 정리한 뒤 이를 검토하고 표현을 다듬는 보조 도구로 사용했다.

본인은 중간발표 이후 교수님 피드백을 바탕으로 fixed stacking ensemble의 한계를 정리했다. 특히 여러 모델의 예측을 고정 weight로 평균하는 방식은 sample별 상황을 반영하기 어렵고, kNN의 k, WLS의 anchor weight, Robust NLS의 residual penalty, ensemble weight가 모두 hand-tuned hyperparameter처럼 보일 수 있다는 문제가 있었다. 이에 따라 최종 알고리즘 방향을 단순 fixed averaging이 아니라, OOF에서 expert별 localization error를 학습하고 sample마다 expected error가 작은 expert에 더 큰 weight를 주는 OOF Error-Gated Mixture-of-Experts 구조로 정했다.

ChatGPT는 이 설계 방향을 검토하고 보고서 형태로 정리하는 데 사용했다. 구체적으로는 중간발표 결과와 교수님 Q&A를 바탕으로 모티베이션을 정리하고, OOF expected-error gating, HGB direct meta learner, Safe Residual Memory의 역할을 자연어와 수식으로 설명하는 데 도움을 받았다. 또한 본인이 선정한 reference 후보들이 실제 코드의 어떤 부분과 연결되는지 확인하고, 각 논문에서 참고한 부분과 본 프로젝트에서 다르게 구현한 부분을 분리해 작성하는 데 활용했다.

Claude Code는 주로 현재 코드의 문제점을 점검하는 용도로 사용했다. 예를 들어 main.py가 hidden test에서 p를 직접 사용하지 않는지, 반환 shape가 (2, num_user)인지, model.pkl 로드가 정상적으로 되는지, 실행 시간이 지나치게 길어질 위험이 있는지 등을 확인하도록 했다. 이 과정에서 KNN 거리 계산의 chunk 크기가 메모리 피크에 영향을 줄 수 있다는 점을 확인했고, 본인이 직접 로컬에서 반복 실행 시간을 측정한 뒤 최종적으로 chunk를 128로 설정했다.

Agent AI 활용 과정에서 본인이 직접 수행한 부분과 AI가 보조한 부분은 다음과 같다.

| 항목 | 본인이 수행한 역할 | Agent AI의 보조 역할 |
|---|---|---|
| 문제 정의 | RTT 기반 실내 측위에서 NLOS, anchor bias, outlier, expert disagreement가 핵심 문제임을 정리했다. | 문제 정의가 보고서 전체에서 일관되게 드러나는지 검토했다. |
| 알고리즘 방향 설정 | fixed stacking의 한계를 분석하고 OOF expected-error gated MoE 구조를 최종 방향으로 정했다. | 교수님 피드백과 연결해 설계 의도를 명확한 문장으로 정리했다. |
| reference 선정 | Mixture-of-Experts, stacking, robust estimation, RANSAC, nearest neighbor, TOA localization 등 코드와 연결되는 논문을 선정했다. | 각 reference의 참고 부분과 본 프로젝트에서 변형한 부분을 구분해 정리했다. |
| 코드 검토 | OOF 결과, 실행 시간, model.pkl, 제출 파일 구성을 직접 확인했다. | shape, hidden p 미사용, chunk 설정, 제출 조건 누락 여부를 점검하는 데 활용했다. |
| 결과 해석 | 여러 OOF 결과 중 최종 report에 사용할 값을 선택하고, hidden test와 OOF 결과를 구분했다. | 성능을 과장하지 않도록 문장 표현과 디스커션을 다듬었다. |

최종적으로 Agent AI는 알고리즘을 자동 생성하는 도구가 아니라, 본인이 수행한 실험과 설계 판단을 검토하고 보고서 품질을 높이는 보조 도구로 사용했다. 최종 알고리즘 구조, reference 연결 방향, OOF 결과 해석, 제출 파일 구성은 본인이 직접 판단했다.

## 4. 결과 도출 & 디스커션

본 절의 결과는 hidden test 결과가 아니라, 제공된 700명 labeled training user에 대해 5-fold OOF 방식으로 계산한 내부 검증 결과이다. 각 fold에서 validation sample을 예측할 때 해당 sample의 정답 좌표는 calibration, expert 학습, gate 학습, meta correction 학습에 직접 사용하지 않았다. 따라서 단순 train-fit 결과보다는 공정한 내부 검증에 가깝지만, 완전히 새로운 hidden dataset에 대한 성능을 보장하지는 않는다.

### 4.1 OOF 결과 요약

| model | mean | median | rmse | p90 | p95 | max | <=1m | <=2m | <=3m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGBR_RobustFeatures | 6.4510 | 5.5781 | 7.6685 | 11.2362 | 14.1200 | 33.6929 | 1.9% | 8.6% | 18.3% |
| Fixed inverse-OOF-error averaging | 6.6609 | 6.2492 | 7.7088 | 10.8887 | 12.6541 | 38.4373 | 1.7% | 6.3% | 15.1% |
| Dynamic expected-error gated MoE | 5.8823 | 5.2327 | 7.0427 | 10.2965 | 12.4595 | 32.3099 | 2.3% | 11.6% | 22.4% |
| OOF final V12A_SafeResidualMemory | 5.5770 | 4.9084 | 6.8143 | 9.9684 | 12.3028 | 32.5841 | 4.3% | 13.9% | 26.0% |
| Oracle best expert upper bound | 3.2620 | 2.7358 | 3.9509 | 6.4090 | 7.3578 | 14.3605 | 13.7% | 36.6% | 53.6% |

위 결과에서 최종 V12A_SafeResidualMemory는 OOF 기준 mean 5.5770 m, median 4.9084 m, RMSE 6.8143 m, p90 9.9684 m를 보였다. 단일 HistGBR expert와 비교하면 mean, RMSE, p90이 모두 개선되었고, Dynamic expected-error gated MoE와 비교해도 meta correction과 residual memory를 추가한 최종 구조가 OOF residual을 더 줄였다.

### 4.2 단계별 개선 해석

| 비교 | mean 변화 | RMSE 변화 | p90 변화 | 해석 |
|---|---:|---:|---:|---|
| HistGBR 단독 → Dynamic gated MoE | 6.4510 → 5.8823 | 7.6685 → 7.0427 | 11.2362 → 10.2965 | 단일 tree expert보다 sample별 expected error에 따라 expert weight를 바꾸는 방식이 유리했다. |
| Fixed inverse-OOF-error averaging → Dynamic gated MoE | 6.6609 → 5.8823 | 7.7088 → 7.0427 | 10.8887 → 10.2965 | 같은 expert들을 단순 고정 평균하는 것보다, sample-adaptive gating이 더 적합했다. |
| Dynamic gated MoE → 최종 V12A | 5.8823 → 5.5770 | 7.0427 → 6.8143 | 10.2965 → 9.9684 | HGB direct meta correction과 Safe Residual Memory가 gated MoE 이후 남은 systematic residual을 추가로 줄였다. |
| HistGBR 단독 → 최종 V12A | 6.4510 → 5.5770 | 7.6685 → 6.8143 | 11.2362 → 9.9684 | robust feature 기반 단일 모델보다 expert 조합, gating, meta correction을 결합한 구조가 더 안정적이었다. |

이 결과는 최종 모델의 성능 향상이 단순히 모델 수를 늘렸기 때문만은 아니라는 점을 보여준다. Fixed inverse-OOF-error averaging은 여러 expert를 사용했지만 mean 6.6609 m로 HistGBR 단독보다도 나빴다. 이는 expert를 많이 넣는 것 자체가 성능을 보장하지 않으며, 실패한 expert의 영향을 sample별로 줄이는 gating이 필요하다는 것을 의미한다. 교수님 피드백에서 지적된 '모델 여러 개를 단순 averaging하지 말고, 상황에 따라 모델을 골라서 써야 한다'는 문제의식이 최종 구조에 직접 반영된 부분이다.

### 4.3 Baseline 비교의 fairness

본 프로젝트에서는 baseline을 크게 세 종류로 구분했다. 첫째, HistGBR_RobustFeatures는 동일한 robust feature를 사용하는 단일 ML expert baseline이다. 둘째, Fixed inverse-OOF-error averaging은 동일한 expert pool을 사용하지만 sample별 gating 없이 OOF 평균 오차 기반으로 고정 결합하는 ensemble baseline이다. 셋째, Dynamic expected-error gated MoE는 meta correction과 residual memory를 적용하기 전의 핵심 gating baseline이다. 이 세 baseline은 모두 같은 5-fold OOF protocol, 같은 labeled training data, 같은 입력 d_hat, 같은 anchor 정보를 사용하므로 최종 V12A와 비교하기에 비교적 공정하다.

반대로 pure triangulation, pure WLS, pure NLS 같은 geometry-only 방법과 최종 ML-gated ensemble을 단순히 동일한 수준의 baseline으로 비교하는 것은 조심해야 한다. Geometry-only solver는 RTT 측정값이 실제 거리와 잘 대응된다는 가정이 강하고, anchor별 NLOS bias나 fingerprint pattern을 충분히 학습하지 못한다. 반면 최종 V12A는 robust feature, tree expert, geometry expert, gate model, meta correction, residual memory를 모두 사용한다. 따라서 deep/ML 계열 모델과 단순 삼각측량을 단순 수치로만 비교하면 모델 복잡도와 사용 정보량이 달라 fair comparison이라고 보기 어렵다. 본 보고서에서는 pure geometry solver를 최종 성능의 주요 비교 대상으로 과장하지 않고, 같은 OOF 절차 안에서 생성된 단일 expert, fixed averaging, dynamic gating, final correction의 ablation 비교를 중심으로 성능을 해석했다.

| 비교 대상 | 최종 모델과 비교가 fair한 정도 | 이유 | 본 보고서에서의 사용 방식 |
|---|---|---|---|
| HistGBR_RobustFeatures | 높음 | 같은 robust feature 기반의 단일 expert이며, OOF protocol이 동일하다. | 단일 ML expert 대비 ensemble/gating의 이득을 확인하는 baseline으로 사용했다. |
| Fixed inverse-OOF-error averaging | 높음 | 같은 expert pool을 사용하지만 sample별 gating이 없는 고정 ensemble이다. | fixed averaging의 한계와 dynamic gating의 필요성을 확인하는 baseline으로 사용했다. |
| Dynamic expected-error gated MoE | 높음 | 최종 모델에서 meta correction과 residual memory만 제거한 구조이다. | 최종 correction layer의 추가 효과를 확인하는 ablation baseline으로 사용했다. |
| Pure WLS / pure NLS / triangulation | 중간~낮음 | 입력 정보와 모델 복잡도, NLOS 대응 능력이 최종 V12A와 다르다. | 직접 경쟁 baseline으로 과장하지 않고, geometry expert의 필요성과 한계를 설명하는 참고 대상으로만 사용했다. |
| Oracle best expert upper bound | 실제 baseline 아님 | sample별 정답을 안다는 가정으로 best expert를 고른 것이므로 hidden test에서 구현 불가능하다. | gate model이 도달할 수 있는 이론적 여지를 보는 upper bound로만 사용했다. |

Oracle best expert upper bound는 mean 3.2620 m로 최종 V12A보다 훨씬 좋지만, 이는 각 sample의 정답을 알고 난 뒤 가장 좋은 expert를 고른 값이다. hidden test에서는 정답을 알 수 없으므로 실제 제출 가능한 baseline이 아니다. 따라서 Oracle 값은 최종 모델과 직접 비교해 성능이 부족하다고만 해석하기보다, gate model이 sample별 best expert를 완벽히 구분하지 못한다는 한계를 보여주는 upper bound로 해석했다.

### 4.4 자체 평가 방식의 fairness와 한계

본 프로젝트의 자체 평가는 5-fold OOF를 기준으로 했다. 각 fold에서 validation sample의 정답 좌표는 해당 fold의 calibration, expert 학습, gate 학습, meta 학습에 직접 사용하지 않았다. 또한 Safe Residual Memory를 OOF에서 평가할 때는 자기 자신의 residual을 그대로 가져오는 leakage를 막기 위해 self-exclusion/LOO-style selection을 적용했다. 따라서 각 sample 기준으로는 자기 자신을 학습에 직접 포함하지 않은 예측을 얻도록 설계했다.

| 평가 요소 | 적용 방식 | fairness 관점의 의미 |
|---|---|---|
| 5-fold OOF expert generation | validation fold를 제외한 train fold로 expert를 학습하고 validation fold를 예측 | train-fit보다 과적합 위험이 낮다. |
| Fold-wise calibration | validation sample의 정답을 calibration parameter 추정에 사용하지 않음 | anchor bias 보정에서 sample-level leakage를 줄인다. |
| Gate model 학습 | OOF expert error를 target으로 사용하고 hidden에서도 계산 가능한 feature만 입력으로 사용 | hidden test에서 사용할 수 없는 실제 error를 직접 쓰지 않는다. |
| Meta correction | OOF prediction과 OOF meta feature를 기반으로 alpha를 선택 | meta layer가 train label을 직접 외우는 위험을 줄인다. |
| Residual memory | OOF 평가에서 self-exclusion/LOO-style residual lookup 사용 | 자기 자신의 residual을 그대로 보정에 쓰는 leakage를 막는다. |
| 최종 성능 주장 | train fit 성능이 아니라 OOF 성능만 사용 | 성능 과장을 줄인다. |

하지만 이 자체 평가에는 한계도 있다. 첫째, OOF는 같은 700명 labeled training set 내부에서 fold를 나눈 평가이므로, 완전히 다른 hidden distribution에 대한 외부 검증은 아니다. 둘째, 여러 OOF run을 반복하면서 가장 좋은 수치를 선택하면 OOF 결과 자체에 맞춘 선택 bias가 생길 수 있다. 이를 줄이기 위해 hidden test 성능을 단정하지 않고, OOF 기준 내부 검증 결과라고 명확히 표현했다. 셋째, 최종 모델은 expert, gate, meta, residual memory가 모두 포함된 복잡한 구조이므로, hidden test에서 anchor 측정 분포나 NLOS 비율이 다르면 OOF보다 성능이 낮아질 수 있다.

### 4.5 Gate 동작 해석

Gate가 어떤 expert를 신뢰했는지도 확인했다. Predicted-best 기준으로는 HistGBR_RobustFeatures가 가장 많이 선택되었고, Calibration_MAP_NLS와 SafeAffine_Trimmed_NLS가 일부 sample에서 보조적으로 선택되었다. Soft weight 기준으로도 HistGBR_RobustFeatures의 평균 weight가 가장 크지만, calibration/NLS/hybrid expert에도 일정 weight가 분산되었다. 즉 최종 모델은 geometry solver 하나에 의존하지 않고, robust fingerprint expert를 중심으로 sample별 보조 expert를 조합하는 구조이다.

| expert | predicted-best count | ratio |
|---|---:|---:|
| ExtraTrees_RobustFeatures | 1 | 0.1% |
| HistGBR_RobustFeatures | 576 | 82.3% |
| Calibration_MAP_NLS | 86 | 12.3% |
| Calibrated_RANSAC_Triplet | 3 | 0.4% |
| SafeAffine_Trimmed_NLS | 34 | 4.9% |

| expert | avg soft weight |
|---|---:|
| KNN_rankdiff_k7_p2 | 0.0053 |
| KNN_rankdiff_k21_p2 | 0.0061 |
| Local_Ridge_KNN40 | 0.0049 |
| ExtraTrees_RobustFeatures | 0.0929 |
| HistGBR_RobustFeatures | 0.4470 |
| MAP_Robust_NLS_KNNPrior | 0.0280 |
| MAP_Trimmed_Residual_NLS | 0.0248 |
| Calibration_MAP_NLS | 0.1249 |
| RANSAC_Triplet_Multilateration | 0.0126 |
| Calibrated_RANSAC_Triplet | 0.0492 |
| Residual_Reranked_Top20_Fingerprint | 0.0587 |
| SafeAffine_MAP_NLS | 0.0699 |
| SafeAffine_Trimmed_NLS | 0.0757 |

이 분포를 보면 gate가 모든 expert를 균등하게 사용하는 것은 아니다. HistGBR_RobustFeatures가 중심 역할을 하고, Calibration_MAP_NLS, SafeAffine_Trimmed_NLS, ExtraTrees, residual-reranked 계열이 보조적인 correction 역할을 한다. 이는 최종 모델이 여러 모델을 무작정 평균한 구조가 아니라, OOF expected error를 바탕으로 expert별 기여도를 다르게 주는 구조임을 보여준다.

### 4.6 실행 시간과 제출 안정성

로컬 환경에서 제공된 700명 입력에 대해 main.py 실행 시간을 세 번 측정한 결과는 다음과 같다. 이 값은 hidden 300명 결과가 아니라, 제공된 700명 파일을 사용한 로컬 실행 기준이다.

| 측정 | 실행 시간 ms | 실행 시간 s |
|---|---:|---:|
| 1회차 | 187811.1640 | 187.8112 |
| 2회차 | 107936.9388 | 107.9369 |
| 3회차 | 107862.5702 | 107.8626 |
| 평균 | 134536.8910 | 134.5369 |
| 중앙값 | 107936.9388 | 107.9369 |

실행 시간은 700명 기준 중앙값 약 107.94초로, 공식 제한 10분보다 낮았다. hidden test는 300명 기준이므로 같은 환경이라면 더 짧게 실행될 가능성이 있다. 최종 코드에서는 KNN 거리 계산 chunk를 128로 설정하여 메모리 피크를 줄였다. 이 chunk 값은 위치추정 알고리즘의 수학적 성능을 바꾸는 hyperparameter라기보다, 한 번에 계산하는 sample 수를 제한하는 실행 안정성 설정이다.

### 4.7 장점, 단점, 향후 개선 방향

본 알고리즘의 장점은 anchor bias와 NLOS outlier를 한 가지 방식으로만 처리하지 않는다는 점이다. Range calibration은 anchor별 systematic bias를 줄이고, robust feature는 거리 패턴을 안정화하며, NLS/RANSAC 계열 expert는 물리적 residual consistency를 반영한다. Error-gated MoE는 expert별 장단점이 sample마다 다르게 나타나는 상황을 반영하고, meta correction과 residual memory는 OOF에서 관찰된 systematic residual을 보정한다.

| 구분 | 내용 |
|---|---|
| 장점 1 | 단일 model이 아니라 fingerprint, geometry, hybrid expert를 함께 사용하여 서로 다른 실패 양상을 보완했다. |
| 장점 2 | fixed averaging이 아니라 OOF expected error 기반 gating을 사용하여 sample별 expert weight를 다르게 적용했다. |
| 장점 3 | anchor별 calibration, robust scale, residual downweighting을 사용해 NLOS/outlier 영향을 줄였다. |
| 장점 4 | Safe Residual Memory에서 self-exclusion/LOO-style selection을 사용해 직접 lookup leakage를 줄였다. |
| 단점 1 | 모델 구조가 복잡하여 hidden distribution이 OOF와 다르면 overfitting 위험이 있다. |
| 단점 2 | NLS, RANSAC, Local Ridge, residual reranking 등 heavy expert가 있어 추론 속도에서 불리할 수 있다. |
| 단점 3 | median error가 공개 상위권보다 높아 typical sample 정확도는 아직 부족하다. |
| 단점 4 | Oracle best expert와의 차이가 크므로 gate model이 sample별 best expert를 완벽히 구분하지 못한다. |

향후 개선 방향은 네 가지이다. 첫째, gate model이 실제 best expert를 더 잘 구분하도록 uncertainty feature와 expert disagreement feature를 개선할 수 있다. 둘째, residual memory가 hidden 분포 변화에 과적합되지 않도록 alpha를 더 보수적으로 조정하거나 confidence가 낮은 sample에서는 correction을 줄이는 방식이 가능하다. 셋째, anchor별 NLOS likelihood를 별도로 추정해 NLS residual weight에 반영하면 p90, p95, max error를 줄일 수 있다. 넷째, heavy expert의 호출 횟수를 줄이거나 일부 expert를 distillation하여 inference speed를 개선할 수 있다.

최종적으로 본 모델은 OOF 기준으로 단일 HistGBR, fixed averaging, base gated MoE보다 개선되었다. 그러나 hidden test 성능은 실제 채점 전에는 알 수 없으며, 본 보고서의 모든 수치는 OOF 기준 내부 검증 결과이다. Hidden test에서는 anchor 측정 분포, NLOS 비율, outlier pattern, scorer 실행 환경에 따라 성능과 속도가 달라질 수 있다.


## 5. Reference

- [1] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural computation, 3(1), 79-87. 이 논문에서 참고한 부분: 여러 expert를 두고 입력 sample에 따라 expert contribution을 다르게 주는 mixture-of-experts 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 원 논문은 neural gating network 중심이지만, 본 프로젝트는 OOF에서 expert별 localization error를 학습하고 predicted expected error가 작을수록 soft weight가 커지는 error-gated MoE로 구현했다.

- [2] Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259. 이 논문에서 참고한 부분: base learner의 out-of-fold prediction을 상위 learner의 입력으로 사용하여 generalization error를 줄이는 stacking 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 단순히 여러 expert prediction을 meta learner에 넣어 좌표를 바로 예측한 것이 아니라, OOF expected-error gating, HGB direct meta learner, Safe Residual Memory를 순차적으로 구성했다.

- [3] Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. Statistical Applications in Genetics and Molecular Biology, 6(1), Article 25. 이 논문에서 참고한 부분: cross-validation 기반으로 여러 candidate learner를 결합하는 Super Learner 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 본 프로젝트는 convex weight만 찾는 방식이 아니라, sample별 expected error prediction을 통해 weight를 다르게 주고, 별도의 direct meta correction과 residual memory를 추가했다.

- [4] Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42. 이 논문에서 참고한 부분: randomization이 강한 tree ensemble을 supervised regression에 사용하는 ExtraTrees 계열 아이디어를 참고했다. / 본 프로젝트에서 다르게 설계한 부분: ExtraTrees를 일반 좌표 회귀에만 사용하지 않고, RTT robust feature 기반 expert와 range error calibration model의 일부로 사용했다.

- [5] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232. 이 논문에서 참고한 부분: gradient boosting을 함수 공간에서의 순차적 근사로 보는 관점을 참고했고, HistGradientBoostingRegressor 기반 nonlinear regression expert와 HGB direct meta learner를 구성했다. / 본 프로젝트에서 다르게 설계한 부분: 일반 tabular regression으로만 사용한 것이 아니라, RTT calibration feature, expert prediction, gate weight, predicted error를 결합한 localization-specific meta feature 위에서 사용했다.

- [6] Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), 381-395. 이 논문에서 참고한 부분: outlier가 포함된 측정값에서 subset 기반 후보 model을 만들고 residual consensus로 평가하는 RANSAC 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 완전한 random sampling이 아니라 anchor triplet 후보를 deterministic하게 만들고, robust residual score로 ranking한 뒤 soft-L1 NLS로 보정하는 RANSAC-like multilateration expert로 변형했다.

- [7] Huber, P. J. (1992). Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution (pp. 492-518). New York, NY: Springer New York. 이 논문에서 참고한 부분: outlier에 덜 민감한 robust estimation 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 원 논문은 location parameter의 robust estimation을 다루지만, 본 프로젝트에서는 anchor별 MAD scale, HuberRegressor 기반 safe affine calibration, soft-L1 NLS, residual downweighting을 통해 RTT NLOS/outlier 영향을 줄이는 데 응용했다.

- [8] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1), 21-27. 이 논문에서 참고한 부분: feature space에서 가까운 sample을 활용하는 nearest-neighbor 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 분류가 아니라 위치 회귀 문제에 적용했고, rank/difference robust feature space에서 inverse-distance weighted KNN, Local Ridge KNN, Safe Residual Memory의 residual interpolation으로 확장했다.

- [9] Cheung, K. W., So, H. C., Ma, W. K., & Chan, Y. T. (2004). Least squares algorithms for time-of-arrival-based mobile location. IEEE transactions on signal processing, 52(4), 1121-1130. 이 논문에서 참고한 부분: 여러 base station의 TOA/range measurement를 이용해 mobile position을 least-squares 방식으로 추정하는 위치추정 문제 설정을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: 본 프로젝트는 RTT 측정값의 NLOS bias와 outlier를 고려하여 fingerprint prior, anchor별 robust calibration, trimmed residual, soft-L1 NLS, ML 기반 gating을 결합했다.

- [10] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830. 이 논문에서 참고한 부분: Python 기반 machine learning 구현 도구인 scikit-learn의 model, preprocessing, cross-validation 구현 체계를 참고했다. / 본 프로젝트에서 다르게 설계한 부분: scikit-learn의 기본 model을 그대로 제출한 것이 아니라, RTT calibration feature, OOF expert generation, expected-error gating, HGB direct meta learner, residual memory를 직접 구성했다.

- [11] Bahl, P., & Padmanabhan, V. N. (2000, March). RADAR: An in-building RF-based user location and tracking system. In Proceedings IEEE INFOCOM 2000. Conference on computer communications. Nineteenth annual joint conference of the IEEE computer and communications societies (Cat. No. 00CH37064) (Vol. 2, pp. 775-784). Ieee. 이 논문에서 참고한 부분: 실내 무선 신호를 위치별 fingerprint로 활용하는 관점을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: RADAR의 RSS radio map을 그대로 구현한 것이 아니라, 제공된 RTT d_hat을 anchor별 거리 패턴 fingerprint로 해석하고, 여기에 range calibration, geometry residual, OOF gating을 결합했다.

- [12] SciPy Developers. (n.d.). scipy.optimize.least_squares. SciPy documentation. 이 문서에서 참고한 부분: Robust NLS에서 사용한 'soft_l1' loss의 수식 rho(z)=2*((1+z)**0.5-1) 및 f_scale에 의한 robust loss scaling 방식을 참고했다. / 본 프로젝트에서 다르게 설계한 부분: SciPy의 일반 least-squares solver를 그대로 적용한 것이 아니라, RTT localization residual, anchor별 robust weight, fingerprint prior, residual downweighting을 결합한 MAP-style robust NLS expert로 사용했다.

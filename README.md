# NeuralODE-NF

![image](https://github.com/user-attachments/assets/c60c6d97-f65b-42d0-964f-0d31c177a995)


라벨/이상 샘플이 없어도” ODE 특성을 활용한 평가 지표(unsupervised validation metric) 제안


### 1) **Solver-Consistency Score (SCS)**: “해석기(스텝/허용오차)에 따라 예측이 얼마나 흔들리나”
- **아이디어**: 같은 입력 윈도우 \(x\)에 대해 ODE를
  - (A) 작은 step / 엄격한 tol
  - (B) 큰 step / 느슨한 tol  
  두 방식으로 풀었을 때 결과(잠재 궤적 \(z(t)\) 또는 복원 \(\hat x\))가 **거의 같아야 정상(매끈한 동역학)**입니다.
- **점수 예시**:  
  - \( \text{SCS}(x)=\|\hat x^{(A)}-\hat x^{(B)}\| \) 또는 \( \int \|z^{(A)}(t)-z^{(B)}(t)\|dt \)
- **왜 이상/비정상에 유리한가**: ODE가 “정상 패턴 manifold”에 잘 맞으면 수치적 변화에 둔감한데, **비정상/분포 밖 입력은 동역학이 더 불안정**해져 solver 민감도가 커지기 쉽습니다.
- **라벨 없이 평가로 쓰는 법**: val에서 평균/상위 분위수(예: P95)를 모니터링 → **감소/안정화**되면 “동역학이 정돈되고 있음” 신호.

---

### 2) **Time-Reversal / Cycle-Consistency Error (TRE/CCE)**: “앞으로 갔다가 돌아오면 원래로 복귀하나”
- **아이디어**: ODE는 연속시간 흐름이라, 설계에 따라 **가역성/사이클 일관성**을 만들기 쉽습니다(예: 인코더→ODE flow→디코더, 혹은 확률흐름 ODE에서 forward/backward).
- **점수 예시**:
  - \( z_0 \xrightarrow{\text{solve}} z_T \xrightarrow{\text{reverse solve}} \tilde z_0 \) 후 \( \|z_0-\tilde z_0\| \)
  - 혹은 \( x \to \hat x \to \tilde x \)의 복귀 오차 \( \|x-\tilde x\| \)
- **왜 유용한가**: 정상 데이터는 학습된 흐름 안에서 “왔다 갔다”가 비교적 잘 맞고, **분포 밖 입력은 cycle 오차가 커지기** 쉽습니다.
- **라벨 없이 평가로 쓰는 법**: val의 TRE 분포가 훈련 중 **낮아지고 안정적**이면 모델이 “정상 동역학을 더 잘 학습” 중이라고 해석 가능.

---

### 3) **Dynamics Residual (DR)**: “관측된 변화율과 ODE가 예측하는 변화율의 불일치”
- **아이디어**: 데이터가 시계열이면 \(dx/dt\)를 유한차분으로 근사해서, 모델의 벡터필드 \(f(x,t)\)와 비교할 수 있습니다.
- **점수 예시**:
  - \( \text{DR}(t)=\left\|\frac{x_{t+\Delta}-x_t}{\Delta}-f(x_t,t)\right\| \), 윈도우 적분/평균
- **장점**: 복원(recon)만 보는 것보다 **“동역학적으로 말이 되는가”**를 직접 측정합니다. 라벨이 없어도 “잘 맞는 모델”을 고르기 좋습니다.

---

### 추가로 ODE에서만(또는 특히) 자연스러운 후보
- **Trajectory length / Action**: \( \int \|f(z,t)\|^2 dt \) (비정상 입력에서 불필요하게 큰 ‘움직임’이 생기면 증가)
- **Stability/Lyapunov proxy**: \( \int \|\partial f/\partial z\| dt \) 또는 Jacobian spectral norm 추정 (민감도/불안정성 측정)
- **(확률흐름 ODE/flow 계열이면) NLL/likelihood 기반 지표**: 라벨 없이도 “정상 분포 적합도”로 체크포인트 선택 가능

---

### 정리: “새 평가 지표”로 제안하기 좋은 형태
논문/보고서 관점에서 제안은 보통 이렇게 잡으면 깔끔합니다.
- **목표**: 라벨 없는 환경에서 모델 선택/학습 모니터링 가능한 지표
- **제안 지표**: SCS + TRE(또는 DR) 같이 “ODE 고유 속성(연속시간/가역성/수치해 안정성)”에 기대는 지표
- **사용법**: val에서 평균/P95 추적, train 대비 분포 안정성(드리프트 감지)까지 같이 보고

---

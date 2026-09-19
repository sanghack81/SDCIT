[![SDCIT Tests](https://github.com/sanghack81/SDCIT/actions/workflows/python-app.yml/badge.svg)](https://github.com/sanghack81/SDCIT/actions)  

SDCIT: Self-Discrepancy Conditional Independence Test
==

개요
-------
`sdcit`는 Lee and Honavar (2017)의 **SDCIT** 알고리즘을 구현하여 파이썬 환경에서 조건부 독립성 검정(Conditional Independence Test)을 수행하는 패키지입니다. 이 알고리즘은 커널 함수를 통해 정의된 관측값 간의 근접성(closeness) 개념과 조건부 치환(conditional permutation)을 활용하여 의사-귀무 표본(pseudo-null sample)을 생성합니다.

이 알고리즘은 효율적인 매칭 최적화를 위해 [`Blossom-V`](https://pub.ista.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz) (Kolmogorov 2009) 라이브러리에 의존합니다. 이 라이브러리는 학술/연구용으로 무료 제공되며, 상업적으로 이용 시 `Blossom-V`의 [상업적 라이선스](https://pub.ista.ac.at/~vnk/software.html)를 별도로 취득해야 합니다.

 
설치 방법
-----
이 패키지는 `python>=3.9` 버전이 필요합니다. 기타 필요한 의존성 패키지는 [requirements.txt](https://github.com/sanghack81/SDCIT/blob/master/requirements.txt)에 명시되어 있습니다. 다음 명령어를 순서대로 실행하면 `SDCIT` 소스코드와 `Blossom-V` 라이브러리를 함께 다운로드하고 설치할 수 있습니다.

```bash
git clone https://github.com/sanghack81/SDCIT
cd SDCIT
# C++ 외부 라이브러리(Blossom-V) 다운로드 및 준비
./setup.sh
# 패키지 빌드 및 로컬 환경 설치
pip install -e .
```

GP 기반 검정(KCIT, FCIT, GP residualization)은 선택 기능이며 gpflow 2.x를 사용합니다.

```bash
pip install -e '.[gp]'
```

### 테스트 실행
모든 모듈이 정상적으로 설치되었는지 확인하기 위해 다음 단위 테스트(Unit Test)를 실행할 수 있습니다.
```bash
pip install pytest pytest-cov
pytest --cov=sdcit sdcit/tests/
```

예제 (Examples)
-----
현재 SDCIT 구현은 최소 8개의 관측을 요구하며, 전체 표본과 반표본의 매칭을 위해
표본 크기가 4의 배수여야 합니다. 커널 행렬들은 크기가 같은 정사각행렬이고
모든 값이 유한해야 합니다. 직접 지정하는 거리 행렬도 유한한 비음수 값을
가져야 합니다. 지원하지 않는 입력은 native 매칭 전에 `ValueError`로 처리합니다.

경험적 upper-tail p-value는 동점을 포함하며
`(1 + count(null >= statistic)) / (number_of_null_draws + 1)`로 계산합니다.

세 가지 간단한 예제를 제공하며, 중앙값 휴리스틱(median heuristic)을 기반으로 커널 행렬을 구하는 코드 조각입니다.

```python
import numpy as np
from sdcit.sdcit_mod import SDCIT
from sdcit.utils import rbf_kernel_median

np.random.seed(0)

N = 200
# 1. 서로 독립인 무작위 변수 3개
X = np.random.randn(N, 2)
Y = np.random.randn(N, 2)
Z = np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)  # 중앙값 휴리스틱(median heuristic) 적용
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))

# 2. 조건부 종속 (conditionally dependent)
# X --> Z <-- Y 구조
Z = X + Y + np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))

# 3. 조건부 독립 (conditionally independent)
# X <-- Z --> Y 구조
Z = np.random.randn(N, 2)
X = Z + np.random.randn(N, 2)
Y = Z + np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))
```

참고 문헌 (References)
-------

> Sanghack Lee, Vasant Honavar **Self-Discrepancy Conditional Independence Test**
> _Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence._ 2017. [Published paper](https://www.auai.org/uai2017/proceedings/papers/16.pdf).

> Gary Doran, Krikamol Muandet, Kun Zhang, and Bernhard Schölkopf. **A Permutation-Based Kernel Conditional Independence Test** 
> _Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence._ 2014.

> Vladimir Kolmogorov. **Blossom V: A new implementation of a minimum cost perfect matching algorithm.**
> In Mathematical Programming Computation (MPC), July 2009, 1(1):43-67.

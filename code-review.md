# AWS / W-LDA 코드 리뷰

## README.md를 읽어보자

#### Install necessary packages
- `mxnet-cu100` (depending CUDA version)
- `matplotlib`
- `scipy`
- `scikit-learn`
- `tqdm`
- `nltk`

#### Pre-process data
- Wikitext-103 [data link](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
- `export PYTHONPATH="$PYTHONPATH:<SOURCE_DIR>"`
- from `SOURCE_DIR`, run `python examples/domains/wikitext103_wae.py`
- `SOURCE_DIR/data/wikitext-103`에 downloading됨

#### Training the model
- from `SOURCE_DIR`, run `./examples/gpu0.sh`

결과값은 `SOURCE_DIR/examples/results`로 저장되고 topic의 상위 단어는 `eval_record.p`에 key `Top Words`와 `Top Words2`로 저장됨. `Top Words2`는 decoder 행렬 가중치의 ranking에 기반한 top words임. `Top Words`는 각 topic의 decoder 출력에 기반한 top words(offset을 더한 decoder행렬의 열과 연관된 토픽).

Note that; NPMI score로 평가하기 위해 분산서버가 `npmi_calc.py`를 실행시킬 것을 요구함. 위 파일은 dictionary와 위키 말뭉치의 역 index 파일을 필요로 함. 현재 해당 파일을 제공하지 않기 때문에 NPMI는 0으로 설정되어 있음. [해당 포스트](https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Evaluate%20Topic%20Models.ipynb)와 같은 글을 참고하여 평가해보시라.

#### License
Apache-2.0 License.

## 소스를 읽어보자

### 구조부터 파악해보자
```
< structure >

awslabs/w-lda
  |ㅡㅡ (dir) examples
  |       |ㅡㅡ (dir) args
  |       |ㅡㅡ (dir) domains
  |       |ㅡㅡ gpu0.sh
  |ㅡㅡ (dir) models
  |       |ㅡㅡ dirichlet.py
  |ㅡㅡ CODE_OF_CONDUCT.md
  |ㅡㅡ CONTRIBUTING.md
  |ㅡㅡ LICENSE
  |ㅡㅡ NOTICE
  |ㅡㅡ compute_op.py
  |ㅡㅡ core.py
  |ㅡㅡ npmi_calc.py
  |ㅡㅡ run.py
  |ㅡㅡ utils.py
```

### py file들 간의 dependencies를 살펴보자

![image](https://user-images.githubusercontent.com/37775784/81651800-e6a2da80-946d-11ea-854b-1d5c1db54c6d.png)

### 모듈의 Flow대로, Top-Down 방식으로 코드를 보자

보는 순서
- `core.py`, `utils.py`
- `dirichlet.py`, `compute_op.py`
- `./examples/`, `run.py`

#### `core.py`
- libraries

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, v_measure_score
import mxnet as mx
from mxnet import gluon, io
import scipy.sparse as sparse
import json
# import misc as nm
# import datasets as nuds
# import wordvectors as nuwe
```

- Objects

```python
class Data:

    """
    :Def:
        ``./examples/~``에서 사용될 기본 객체!

    :Role:
        path로부터 data, labels, maps를 읽고
        minibatch를 출력하는 loader와 iter를 생성,
        (gluon, io를 활용하여)
        iter 초기화 메서드와 y/y_label 시각화 staticmethod도 존재.
    """

    def __init__(self,
                 batch_size=1,
                 data_path='',
                 ctx=mx.cpu(0)):
        """
        Data Constructor

        1. data_path를 인자로 load 메서드를 실시,
          data, labels, maps를 받아옴
        2. 위 데이터를 기반으로 ``dict(zip(names, datas))`` 테크닉으로 사전 생성
        3. batch_size만큼 데이터를 쌓음(tile)
        4. maps도 2와 같이 수행
        5. mxnet의 gluon 혹은 io의 DataLoader/NDArrayIter 객체를 data, batch별 호출
           이를 iter(dl)로 binding
        6. 5에서 dls를 dataloaders로, dis를 dataiters로 저장
        """
        pass

    def dataloader(self, data, batch_size, shuffle=True):
        """
        data의 mini-batch를 생성하기 위한 data loader를 구축

        - data가 None이면 None을 반환
        - 아닐 때 data가 ndarray인지 아닌지에 따라 아래 행동 결정
            - True: gluon.data.DataLoader 객체 반환
            - False: io.NDArrayIter 객체 반환
        """
        pass

    def fore_reset_data(self, key, shuffle=True):
        """
        epoch이 재시작할 때 minibatch index를 0으로 초기화.

        - self.data[key]가 None이 아닐 경우만 아래 내역을 실시
        - data[key]가 ndarray일 경우 dataloader와 iter 할당,
        - 아닐 경우 dataiters만 hard_reset() 실시.
        - wasreset[key]에 True 할당.
        """
        pass

    def minibatch(self, key, pad_width=0):
        """
        self.ctx device에 저장된 데이터의 minibatch를 반환
        세부 detail 생략.
        """
        pass

    def get_documents(self, key, split_on=None):
        """
        `self.data`에 해당하는 documents의 minibatch를 검색.
        """
        pass

    @staticmethod
    def visualize_series(y, ylabel, file, args, iteration, total_samples, labels=None):
        """
        y vs iterations and epochs를 그리고 figure를 저장.
        대강 확인함. 나중에 사용해서 직접 그려보기.
        """
        pass

    def load(self, path=''):
        """
        path로부터 data와 maps 읽어오기.
        """
        pass
```

```python
class ENet(gluon.HybridBlock):

    """
    A gluon HybridBlock Encoder (skeleton) class.
    """

    def __init__(self):
        """ Constructor for Encoder """
        super().__init__()

    def hybrid_forward(self, x):
        """ Encodes x """
        raise NotImplementedError('상속해서 method 작성해라.')

    def init_weights(self, weighs=None):
        """
        encoder weights 초기화. Default는 Xavier 초기화

        웃긴게, self.weight_files setting도 안하곤 찾는다. super에 있겠지.

        > weights_file이 ''이 아니고 weights가 None이면 해당 path에서 loading.
        > 위에서 실패하면 `load_params`메서드로 읽는게 아니라 pickle로 읽음.

        > 위 블럭을 지나 weights가 None이 아니라면 model의 param을 받은 weight로
        > 초기화시킨다. freeze 조건 탐색해서 lr_mult = 0.으로 만들거나.

        위 두 블럭에 안걸렸다면, 파라미터를 Xavier로 초기화.
        """
        pass

    def freeze_params(self):
        """
        parameter freeze
        """
        for p in self.collect_params().values():
            p.lr_mult = 0.
```

```python
class DNet(gluon.HybridBlock):

    """
    A gluon HybridBlock Dncoder (skeleton) class.
    """

    def __init__(self):
        """ Constructor for Dncoder """
        super().__init__()

    def hybrid_forward(self, y, z):
        """ Dncodes x """
        raise NotImplementedError('상속해서 method 작성해라.')

    def init_weights(self, weighs=None):
        """
        Dncoder weights 초기화. Default는 Xavier 초기화

        ENet skeleton 코드와 동일하게 진행.
        """
        pass

    def freeze_params(self):
        """
        ENet skeleton 코드와 동일하게 진행.
        """
        pass
```

```python
class Compute:

    """
    학습, 테스팅, outputs retrieving을 관리할 skeleton class.
    ``flesh``를 위해 ``compute_op.py``를 볼 것.
    """

    def __init__(self, data, Enc, Dec, Dis_y, args):
        """
        연산 관리자!

        1. data, Enc, Dec, Dis_y, args, model_ctx, ndim_y 기록
        2. args['optim']이 Adam/Adadelta/RMSprop/SGD인지에 따라
           optim_{enc/dec/dis_y}를 각 weight로 설정!
           그리고 이를 기록

        토치로 짤 때도 이런식으로 저장해두면 편리하겠다!
        함수형으로만 짜지 말고.
        """
        pass

    def train_op(self):
        """ minibatch data를 사용하여 model을 학습 """
        return None, None, None, None

    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        """ num_samples를 사용하여 model을 평가 """
        if num_samples is None:
            num_samples = self.data.data[dataset].shape[0]
        if reset:
            # Reset Data to Index Zero
            self.data.force_reset_data(dataset)
            self.data.force_reset_data(dataset+'_with_labels')
        return None, None, None, None

    def get_outputs(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        """ model로부터 num_samples만큼의 raw outputs을 retrieve """
        if num_samples is None:
            num_samples = self.data.data[dataset].shape[0]
        if reset:
            # Reset Data to Index Zero
            self.data.force_reset_data(dataset)
            self.data.force_reset_data(dataset+'_with_labels')
        return None, None, None, None, None, None
```

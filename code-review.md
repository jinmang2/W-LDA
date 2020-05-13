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

#### `./models/dirichlet.py`

- libraries

```python
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from scipy.special import logit, expit
```

- dependencies

```python
from core import ENet, DNet
```

- Objects

```python
class Encoder(ENet):

    """ A gluon HybridBlock Encoder class """

    def __init__(self,
                 model_ctx: mx.ctx,
                 batch_size: int,
                 input_dim: int,
                 n_hidden: Union[int, List]=64,
                 ndim_y: int=16,
                 ndim_z: int=10,
                 n_layers: int=0,
                 nonlin: Optional[str]=None,
                 weights_files: str='',
                 freeze: bool=False,
                 latent_nonlin: str='sigmoid',
                 **kwargs):
        """ Constructor for encoder """
        super().__init__()
        # 왜 이렇게 읽는지 이해가 안된다.
        # 논문 읽거나 차후 torch로 구현할 때 수정해야지.
        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hiddens[0]
            n_hidden = n_layers * [n_hidden]
        else:
            n_layers = len(n_hidden)

        # non-linearity setting.
        if nonlin == '': nonlin = None

        # torch는 nn.Module이 아래 작업을 처리해줌.
        # name_scope는 ``mxnet.gluon.block``의 `_BlockScope`
        # context-manager를 호출. name space 관리를 해줌.
        # prefix 및 hint, 기타 profiler 기능도 수행하는 듯.
        in_units = input_dim
        with self.name_scope():
            # Sequential 호출. keras의 그 것과 유사
            self.main = nn.HybridSequential(prefix='encoder')
            # Encoder는 n_layer만큼의 Dense(torch의 Linear)를 가짐
            # 처음 layer만 input_dim, 그 다음엔 n_hidden[i]로!
            # activation is `nonlin`
            for i in range(n_layers):
                self.main.add(
                    nn.Dense(
                        n_hidden[i],
                        in_units=in_units,
                        activation=nonlin
                    )
                )
                in_units = n_hidden[i]
            # 마지막 linear는 ndim_y로 빠져나올 수 있게 setting.
            # activation is None
            self.main.add(
                nn.Dense(
                    ndim_y,
                    in_units=in_units,
                    activation=None
                )
            )

        self.model_ctx = model_ctx
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.ndim_y = ndim_y # 아니, 쓰지도 않을거면서 ㄷㄷ
        self.ndim_z = ndim_z # 아니, 쓰지도 않을거면서 ㄷㄷ
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze
        self.dist_params = [None]

    def hybrid_forward(self, F, x):
        """ F는 ``mxnet.nd`` or ``mxnet.sym``로 passing된다고 하네. """
        y = self.main(x)
        return y
```

```python
class Decoder(DNet):

    """ A gluon HybridBlock Decoder class with Multinomial likelihood, p(x|z)"""

    def __init__(self,
                 model_ctx: mx.ctx,
                 batch_size: int,
                 output_dim: int,
                 n_hidden: Union[int, List]=64,
                 ndim_y: int=16,
                 ndim_z: int=10,
                 n_layers: int=0,
                 nonlin: str='sigmoid',
                 weights_files: str='',
                 freeze: bool=False,
                 latent_nonlin: str='sigmoid',
                 **kwargs):
        """ Constructor for decoder """
        super().__init__()
        # Encoder와 동일.
        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hiddens[0]
            n_hidden = n_layers * [n_hidden]
        else:
            n_layers = len(n_hidden)

        # Encoder와 동일
        if nonlin == '': nonlin = None

        ### Caution!!! Encoder와 다름.
        in_units = n_hidden[0]
        with self.name_scope():
            self.main = nn.HybridSequential(prefix='decoder')
            # 이럴거면 n_layers를 왜 받는거냐 ㄷㄷ
            # decoder는 하나만!
            self.main.add(
                nn.Dense(
                    n_hidden[0],
                    in_units=ndim_y,
                    activation=None
                )
            )

        self.model_ctx = model_ctx
        self.n_hidden = n_hidden
        self.ndim_y = ndim_y # 아니 안쓸거면서 ㄷㄷ
                             # 아 Decoder에선 쓰넹 ㅇㅅㅇ
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze

    def hybrid_forward(self, F, y):
        out = self.main(y)
        return out

    def y_as_topic(self, eps=1e-10):
        y = np.eye(self.ndim_y)
        return mx.nd.array(y)
```

```python
class Discriminator_y(ENet):

    """
    A gluon HybridBlock Discriminator Class for y
    """

    def __init__(self,
                 model_ctx: mx.ctx,
                 batch_size: int,
                 output_dim: int=2,
                 n_hidden: Union[int, List]=64,
                 ndim_y: int=16,
                 n_layers: int=0,
                 nonlin: str='sigmoid',
                 weights_files: str='',
                 freeze: bool=False,
                 latent_nonlin: str='sigmoid',
                 apply_softmax: bool=False,
                 **kwargs):
        """ Constructor for Discriminator Class for y """
        super().__init__()
        # Encoder와 동일.
        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hiddens[0]
            n_hidden = n_layers * [n_hidden]
        else:
            n_layers = len(n_hidden)

        # Discriminator_y에서 새로 추가된 부분.
        if latent_nonlin != 'sigmoid':
            print('NOTE: Latent z will be fed to decoder in logit-space (-inf, inf).')
        else:
            print('NOTE: Latent z will be fed to decoder in probability-space (0, 1).')

        # Encoder와 동일.
        if nonlin == '': nonlin = None

        # Encoder와 동일.
        in_units = ndim_y
        with self.name_scope():
            self.main = nn.HybridSequential(prefix='discriminator_y')
            for i in range(n_layers):
                self.main.add(
                    nn.Dense(
                        n_hidden[i],
                        in_units=in_units,
                        activation=nonlin
                    )
                )
                in_units = n_hidden[i]
            self.main.add(
                nn.Dense(
                    output_dim,
                    in_units=in_units,
                    activation=None
                )
            )

        self.model_ctx = model_ctx
        self.n_hidden = n_hidden
        self.ndim_y = ndim_y
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze
        self.apply_softmax = apply_softmax

    def hybrid_forward(self, F, y):
        logit = self.main(y)
        if self.apply_softmax:
            return F.softmax(logit)
        return logit
```

#### `utils.py`

- libraries

```python
import socket
import pickle
import argparse
import time
import os
from functools import reduce
import numpy as np
from scipy.special import logit
import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
from tqdm import tqdm
import mxnet as mx
# from sklearn.neighbors import NearestNeighbors
# from IPython import embed
import collections
```

- functions

```python
"""
아래는 torch에서 굉장히 쉽게 처리 가능.
"""
def gpu_helper(gpu):
    if gpu >= 0 and gpu_exists(gpu):
        model_ctx = mx.gpu(gpu)
    else:
        model_ctx = mx.cpu()
    return model_ctx

def gpu_exists(gpu):
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(gpu))
    except:
        return False
    return True
```
```python
"""
아래도 torch로 매우 쉽게 처리 가능
"""
def reverse_dict(X, xnew):
    for i in range(len(X)):
        if isinstance(xnew[i], list):
            X[i] = stack_numpy(X[i], xnew[i])
        else:
            X[i] = np.vstack([X[i], xnew[i]])
    return X

def to_numpy(X):
    x_npy = []
    for x in X:
        if isinstance(x,list):
            x_npy += [to_numpy(x)]
        else:
            x_npy += [x.asnumpy()]
    return x_npy

def stack_numpy(X,xnew):
    for i in range(len(X)):
        if isinstance(xnew[i],list):
            X[i] = stack_numpy(X[i], xnew[i])
        else:
            X[i] = np.vstack([X[i], xnew[i]])
    return X
```

```python
"""
``run.py``에서 필요한 method.
보니, copyto, collect_params 등으로 미루어 보아,
어디서 활용하는지를 봐야 코드 이해가 쉬울 듯.
``yz_as_topics``와 같이 띄용한 인자도 튀어나오기도 하고.
"""
def get_topic_words_decoder_weights(D,
                                    data,
                                    ctx,
                                    k=10,
                                    decoder_weights=False):
    pass


def get_topic_words(D, data, ctx, k=10):
    pass


def calc_topic_uniqueness(top_words_idx_all_topics):
    pass


def request_pmi(topic_dict=None, filename='', port=1234):
    pass


def print_topics(topic_json,
                 npmi_dict,
                 topic_uniqs,
                 data,
                 print_topic_names=False):
    pass


def print_topic_with_scores(topic_json, **kwargs):
    pass
```

#### `compute_op.py`

- libraries

```python
import numpy as np

from tqdm import tqdm

from mxnet import nd, autograd, gluon, io

# from diff_sample import normal
import os
```

- dependencies

```python
from core import Compute
from utils import to_numpy, stack_numpy
```

- functions

```python
def mmd_loss(x, y, ctx_model, t=0.1, kernel='diffusion'):
    """
    information diffusion kernel을 가진 mmd loss 계산
    :param x: batch_size x latent dimension

    eps = 1e-6
    n:= 'batch_size', d:= 'latent dimension'

    kernel이 tv냐 diffusion 혹은 그 외의 것이냐로 나눠짐.
    구할 것은 ``x_loss``, ``y_loss``, ``xy_loss``.
    (1) kernel == 'tv'
        x = [[ x11 x12 ... x1d ]  = [[ r1 ]
             [ x21 x22 ... x2d ]  =  [ r2 ]
             [  :   :   :   :  ]  =  [ :  ]
             [ xn1 xn2 ... xnd ]] =  [ rn ]] 일 때,
        i \in (1, n)에 대하여 nC2 경우의 수만큼 L1 loss를 계산. 즉,
        x_loss = {
            l1(r1, r2) + l1(r1, r2) + ... + l1(r1, rn-1) + l1(r1, rn)
                       + l1(r2, r3) + ... + l1(r2, rn-1) + l1(r2, rn)
                       ...
                                                         + l1(rn-1, rn)
        }
        x_loss /= n(n-1) (경우의 수만큼 나눠줌. 1/2는 남김.)

        y = [[ y11 y12 ... y1d ]  = [[ s1 ]
             [ y21 y22 ... y2d ]  =  [ s2 ]
             [  :   :   :   :  ]  =  [ :  ]
             [ yn1 yn2 ... ynd ]] =  [ sn ]] 일 때,

        i \in (1, m)에 대하여 mC2 경우의 수만큼 L1 loss를 계산. 즉,
        y_loss = {
            l1(s1, s2) + l1(s1, s2) + ... + l1(s1, sm-1) + l1(s1, sm)
                       + l1(s2, s3) + ... + l1(s2, sm-1) + l1(s2, sm)
                       ...
                                                         + l1(sm-1, sm)
        }
        y_loss /= m(m-1)

        x와 y의 차이로도 L1 norm을 계산. 총 n x m만큼의 값을 합산. 즉,
        xy_loss = {
            l1(r1, s1) + l1(r1, s2) + ... + l1(r1, sm) +
            l1(r2, s1) + l1(r2, s2) + ... + l1(r2, sm) + ...
            l1(rn, s1) + l1(rn, s2) + ... + l1(rn, sm) +
        }
        xy_loss /= nm

        % 위의 L1Norm은 sum{abs{array}}임.

    (2) kernel != 'tv', that is, kernel == 'diffusion'
        위의 x, y에 대하여, n == m.
        xx = qx qx^T :math: \in \mathbb{R}^{n \times n}
        yy = qy qy^T :math: \in \mathbb{R}^{m \times m}
        xy = qx qy^T :math: \in \mathbb{R}^{n \times m}
            where qx = sqrt{clip{x, 1e-6, 1}}
                  qy = sqrt{clip{y, 1e-6, 1}}
        set off-diagonal matrix as following;
            off_diag = [[0, 1, 1, ..., 1, 1]
                        [1, 0, 1, ..., 1, 1]
                        [:, :, :, ..., :, :]
                        [1, 1, 1, ..., 0, 1]
                        [1, 1, 1, ..., 1, 0]]
        let tmpt == t, dim = d - 1, a_{X} = clip{{X}, 0, 1-eps}.
        Using diffusion kernel as follow, calc three kernel result.
            difussion_kernel>
                :math: (4\pi tmpt)^{\dim/2} * \exp(\arccos(a)^2 / tmpt)
            or, :math: \exp(\arccos(a)^2 / tmpt)
            k_xx = diffusion_kernel(a_xx, tmpt, dim)
            k_yy = diffusion_kernel(a_yy, tmpt, dim)
            k_xy = diffusion_kernel(a_xy, tmpt, dim)
        Then, off-diag 항들을 모두 더함.
        x_loss  = sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
        y_loss  = sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
        xy_loss = sum_xy = (k_xy * off_diag).sum() / (n * (n-1))
    (1) 혹은 (2)를 통해 얻은 x_loss, y_loss, xy_loss를 아래와 같이 연산.
    total_loss = x_loss + y_loss - xy_loss
    total_loss를 반환.
    """
```

- objects

```python
# Compute는 core에 종속.
class Unsupervised(Compute):

    def __init__(self, data, Enc, Dec, Dis_y, args):
        super().__init__(data, Enc, Dec, Dis_y, args)

    def unlabeled_train_op_mmd_combine(self, update_enc=True):
        """
        Train the MMD model

        First, Retrieve data, docs
            using ``data.get_documents(key='train')`` methods

        Second, dirichlet 분포에서 y_true를 sampling
            y_true.shape == (ndim_y, batch_size)

        Third, 아래 연산 과정을 수행
            with autograd.record():
                y_u = Softmax(Enc(docs)) # Enc(docs)를 y라 하자.
                if a:=latent_noise > 0:
                    y_noise를 dirichlet 분포에서 sampling하고
                    y_u = (1-a)*y_u + a*y_noise
                x_reconstruction_u = Dec(y_u)
                logit = log_Softmax(x_reconstruction_u)
                loss_reconstruction = mean(sum(-docs * logit, axis=1))
                loss_total = loss_reconstruction * recon_alpha

                ### MMD Phase ###
                if there exists ``adverse``,
                    y_fake = Softmax(Enc(docs))
                    loss_mmd = mmd_loss(y_true, y_fake, t=kernel_alpha)
                    loss_total += loss_mmd
                if there exists ``l2_alpha``,
                    loss_total += l2_alpha * mean(sum(square(y_u), axis=1))

                loss_total.backward() # 오차 역전파
            optimizer_enc.step(1)
            optimizer_dec.step(1)

        Fourth, calc latent values as follow;
            latent_max = zeros(ndim_y)
            for max_ind in argmax(y, axis=1):
                latent_max[max_ind] += 1.0
            latent_max /= batch_size
            latent_entropy = mean(sum(-y_u*log(y_u+eps), axis=1))
            latent_v = mean(y_u, axis=0)
            dirich_entropy = mean(sum(-y_true*log(y_true+eps), axis=1))

        Fifth, calc loss_mmd_return
            if \exist ``adverse``,
                loss_mmd_return = loss_mmd.asscalar()
            else, loss_mmd_return = 0.0

        Returns below.
            (
                mean(loss_reconstruction).asscalar(),
                loss_mean_return,
                latent_max.asnumpy(),
                latent_entropy.asscalar(),
                latent_v.asnumpy(),
                dirich_entropy.asscalar(),
            )
        """
        pass

    def retrain_enc(self, l2_alpha=0.1):
        """
        Re-train.

        ``unlabeled_train_op_mmd_combine``를 아주 간략하게만 계산.

        1. docs retrieve
        2. 아래 계산만 수행
            with autograd.record():
                y = Enc(docs)
                y_u = Softmax(y)
                x_reconstruction_u = Dec(y_u)
                logits = log_Softmax(x_reconstruction_u)
                loss_reconstruction = mean(sum(-docs*logits, axis=1))
                loss_reconstruction += l2_alpha * mean(l1norm(y_u, axis=1))
                loss_reconstruction.backward()
            optimizer_enc.step(1) # Enc Optim만 학습.
            return loss_reconstruction.asscalar()
        """
        pass

    def unlabeled_train_op_adv_combine_add(self, update_enc=True):
        """
        Trains the GAN model

        First, settings
            eps = 1e-10
            docs = data.get_documents(key='train') # Retrieve data
            class_true  = zeros(batch_size)
            class_fakse = zeros(batch_size)
            loss_reconstruction = zeros(1)

        Second, ### Adversarial phase ###
            discriminator_z_confidonce_true = zeros(1)
            discriminator_z_confidonce_fake = zeros(1)
            discriminator_y_confidonce_true = zeros(1)
            discriminator_y_confidonce_fake = zeros(1)
            loss_discriminator = zeros(1)
            dirich_entropy = zeros(1)

        Third, ### Generator phase ###
            loss_generator = zeros(1)

        Fourth, 아래 연산 과정을 수행
            ### Reconstruction phase ###
            with autograd.record():
                y = Enc(docs)
                y_u = Softmax(y)
                x_reconstruction_u = Dec(y_u)
                logits = log_Softmax(x_reconstruction_u)
                loss_reconstruction = sum(-docs*logits, axis=1)
                loss_total = loss_reconstruction * recon_alpha

                if there exists ``adverse``, # and np.random.rand() < .8
                    y_true를 sampling, shape is (ndim_y, batch_size)
                    dy_true = Dis_y(y_true)
                    dy_fake = Dis_y(y_u)
                    discriminator_y_confidonce_true = mean(softmax(dy_true)[:, 0])
                    discriminator_y_confidonce_fake = mean(softmax(dy_fake)[:, 1])
                    loss_discriminator = SoftmaxCEL(dy_true, class_true) +
                                         SoftmaxCEL(dy_fake, class_fake)
                    loss_generator = SoftmaxCEL(dy_fake, class_true)
                    loss_total += loss_discriminator + loss_generator
                    dirich_entropy = mean(sum(-y_true*log(y_true+eps), axis=1))
            loss_total.backward()

        Fifth, Updating optimizers
            optimizer_enc.step(batch_size)
            optimizer_dec.step(batch_size)
            optimizer_dis_y.step(batch_size)

        Sixth, calc latent values as follow;
            latent_max = zeros(ndim_y)
            for max_ind in argmax(y_u, axis=1): # 어라 위랑 다르네
                latent_max[max_ind] += 1.0
            latent_max /= batch_size
            latent_entropy = mean(sum(-y_u*log(y_u+eps), axis=1))
            latent_v = mean(y_u, axis=0)

        Returns (
            nd.mean(loss_discriminator).asscalar(),
            nd.mean(loss_generator).asscalar(),
            nd.mean(loss_reconstruction).asscalar(),
            nd.mean(discriminator_z_confidence_true).asscalar(),
            nd.mean(discriminator_z_confidence_fake).asscalar(),
            nd.mean(discriminator_y_confidence_true).asscalar(),
            nd.mean(discriminator_y_confidence_fake).asscalar(),
            latent_max.asnumpy(),
            latent_entropy.asscalar(),
            latent_v.asnumpy(),
            dirich_entropy.asscalar()
        )
        """

    def test_synthetic_op(self):
        """ 간단히 testing !! """
        pass

    @overrides
    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        """ return unlabeled_loss, labeled_loss, labeled_acc """
        pass

    def save_latent(self, saveto):
        """ 저장하는 코드, 내가 따로 작성하면 됨. """
        pass
```

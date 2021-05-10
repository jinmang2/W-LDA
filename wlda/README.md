# W-LDA

## Development Note

### Config 관리
- hydra, omegaconf 등 configuration 관리 모듈 서치
- huggingface의 `hfArgumentParser` 객체를 활용하여 config 파일을 관리
- shell, python으로 config 실행 가능

### Data, Compute 관리
- Pytorch Lightning의 Trainer class를 참고하여 개발

### Dependencies
- numpy
- torch
- dataclasses
- scipy
- nltk
- scikit-learn

### Reference design
- transformers
- datasets

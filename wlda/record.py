import json
import copy
from overrides import overrides
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any


class JSONSaveLoadMixin:
    def save_to_json(self, json_path: str):
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


class ReprMixin:
    def __repr__(self):
        cls_name = self.__class__.__qualname__
        FIELDS = self.__dataclass_fields__.values()
        contents = ", ".join(
            [field.name + ": " + str(field.type).split(".")[-1] 
            for field in FIELDS]
        )
        return cls_name + "(" + contents + ")"


class RecordManager:
    report = {}

    def record(self, slot: str, value: Any):
        self.report.update({slot: value})

    def update(self, report: Optional[Dict] = None):
        if report is None:
            report = self.report
        for slot, value in report.items():
            if slot in self.__dataclass_fields__.keys():
                getattr(self, slot).append(value)
        self.reset()
    
    def reset(self):
        self.report = {}


@dataclass(repr=False)
class TrainRecorder(JSONSaveLoadMixin, ReprMixin, RecordManager):
    loss_discriminator: List[float] = field(default_factory=list)
    loss_generator: List[float] = field(default_factory=list)
    loss_reconstruction: List[float] = field(default_factory=list)
    latent_max_distr: List[float] = field(default_factory=list)
    latent_avg_entropy: List[float] = field(default_factory=list)
    latent_avg: List[float] = field(default_factory=list)
    dirich_avg_entropy: List[float] = field(default_factory=list)
    loss_labeled: List[float] = field(default_factory=list)

    @overrides
    def record(self, slot: str, value: float):
        self.report[slot] += value

    @overrides
    def reset(self):
        field_names = self.__dataclass_fields__.keys()
        self.report = {f_name: 0. for f_name in field_names}

    __post_init__ = reset


@dataclass(repr=False)
class EvalRecorder(JSONSaveLoadMixin, ReprMixin, RecordManager):
    npmi: List[float] = field(default_factory=list)
    topic_uniqueness: List[float] = field(default_factory=list)
    top_words: List[float] = field(default_factory=list)
    npmi2: List[float] = field(default_factory=list)
    topic_uniqueness2: List[float] = field(default_factory=list)
    top_words2: List[float] = field(default_factory=list)
    u_loss_train: List[float] = field(default_factory=list)
    l_loss_train: List[float] = field(default_factory=list)
    u_loss_val: List[float] = field(default_factory=list)
    l_loss_val: List[float] = field(default_factory=list)
    u_loss_test: List[float] = field(default_factory=list)
    l_loss_test: List[float] = field(default_factory=list)
    l_acc_train: List[float] = field(default_factory=list)
    l_acc_val: List[float] = field(default_factory=list)
    l_acc_test: List[float] = field(default_factory=list)

    @overrides
    def record(self, slot: str, value: float):
        self.report[slot] += value

    @overrides
    def reset(self):
        field_names = self.__dataclass_fields__.keys()
        self.report = {f_name: 0. for f_name in field_names}

    __post_init__ = reset
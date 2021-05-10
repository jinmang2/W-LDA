from overrides import overrides
from .trainer import Trainer


def summarization_metrics(pred: EvalPrediction) -> Dict:
    pred_str = tokenizer.batch_decode(
        pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(
        pred.label_ids, skip_special_tokens=True)
    pred_str = list(map(str.strip, pred_str))
    label_str = list(map(str.strip, label_str))
    rouge = caculate_rouge(pred_str, label_str)
    summ_len = np.round(np.mean(list(map(non_pad_len, pred.predictions))), 1)
    rouge.update({"gen_len": summ_len})
    return rouge


class UnsupervisedTrainer(Trainer):
    """
    Unsupervised Trainger class for W-LDA
    """

    def compute_loss(self, model, inputs):
        """ computes the mmd loss with information diffusion kernel """
        pass


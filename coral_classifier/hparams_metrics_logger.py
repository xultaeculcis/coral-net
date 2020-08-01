import argparse
from typing import Union, Dict, Any, Optional

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard.summary import hparams


class HparamsMetricsTensorBoardLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        super(HparamsMetricsTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, params: Union[Dict[str, Any], argparse.Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self,
                                params: Union[Dict[str, Any], argparse.Namespace],
                                metrics: Dict[str, Any]) -> None:
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)

        self.hparams.update(params)
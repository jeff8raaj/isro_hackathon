from .metrics import RemoteSensingMetrics


class Evaluator:

    def __init__(self):
        self.metrics = RemoteSensingMetrics()

    def evaluate(self, pred, target):
        return {
            "RMSE": self.metrics.rmse(pred, target).item(),
            "PSNR": self.metrics.psnr(pred, target),
            "SSIM": self.metrics.ssim(pred, target),
            "SAM": self.metrics.sam(pred, target).item()
        }

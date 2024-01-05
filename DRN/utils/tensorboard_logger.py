from torch.utils.tensorboard import SummaryWriter
import os
os.environ['KMP_WARNINGS'] = '0'


class TensorboardLogger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
import torch
import torch.nn as nn

# Adapted from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class RunningStatisticsNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(RunningStatisticsNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats=True)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            # use biased var in train
            var = input.var([0], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = (exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var = (exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var)

        # input = (input - self.running_mean[None, :]) / (torch.sqrt(self.running_var[None, :] + self.eps))
        input = input - self.running_mean[None, :]
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        print(f"running_mean: {self.running_mean}")

        return input

# class RunningMeanBatchNorm1d(nn.BatchNorm1d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__(num_features, eps=eps, momentum=momentum)
#
#     def forward(self, x):
#         # Always use stored running mean/var, even during training
#         return nn.functional.batch_norm(
#             x, self.running_mean, self.running_var, self.weight, self.bias,
#             training=False, eps=self.eps
#         )
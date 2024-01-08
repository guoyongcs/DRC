import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class FocalMSE(nn.Module):
	def __init__(self):
		super(FocalMSE, self).__init__()
		self.LAMBDA = 0

	def forward(self, input, target):
		assert len(input.size())==4, 'dimension error'
		diffL1 = F.l1_loss(input, target, size_average=False, reduce=False)
		diffL1 = torch.pow(diffL1, self.LAMBDA)
		diffL1 = diffL1.view(input.size(0), -1)
		alpha = F.softmax(diffL1, 1).view_as(input)/input.size(0)
		diffL2 = F.mse_loss(input, target, size_average=False, reduce=False)
		output = torch.sum(alpha * diffL2)
		return output
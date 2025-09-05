import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def _select_device(device: str = "auto") -> torch.device:
	if device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device)


def _ensure_dir(path: str) -> None:
	if path and not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


class VGGFeatures(nn.Module):
	"""Extract feature activations at specified layer indices from VGG-19."""

	def __init__(self, content_indices: List[int], style_indices: List[int]):
		super().__init__()
		vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

		layers = []
		for layer in vgg.children():
			# Disable in-place ReLU to keep gradients stable for optimization on pixels
			if isinstance(layer, nn.ReLU):
				layers.append(nn.ReLU(inplace=False))
			else:
				layers.append(layer)

		self.features = nn.Sequential(*layers).eval()
		for p in self.features.parameters():
			p.requires_grad = False

		self.content_indices = set(content_indices)
		self.style_indices = set(style_indices)

	def forward(self, x: torch.Tensor) -> Tuple[dict, dict]:
		content_feats, style_feats = {}, {}
		for idx, layer in enumerate(self.features):
			x = layer(x)
			if idx in self.content_indices:
				content_feats[idx] = x
			if idx in self.style_indices:
				style_feats[idx] = x
		return content_feats, style_feats


def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
	batch, channels, height, width = feat.size()
	feat = feat.view(batch, channels, height * width)
	gram = torch.bmm(feat, feat.transpose(1, 2))
	return gram / (channels * height * width)


class NeuralStyleTransfer:
	"""High-level API for neural style transfer using VGG-19 features."""

	def __init__(
		self,
		device: str = "auto",
		image_size: int = 512,
		content_weight: float = 1.0,
		style_weight: float = 1_000_000.0,
		tv_weight: float = 1.0,
		content_layers: Optional[List[int]] = None,
		style_layers: Optional[List[int]] = None,
	):
		self.device = _select_device(device)
		self.image_size = image_size
		self.content_weight = content_weight
		self.style_weight = style_weight
		self.tv_weight = tv_weight

		self.content_indices = content_layers or [21]
		self.style_indices = style_layers or [0, 5, 10, 19, 28]

		self.feature_extractor = VGGFeatures(self.content_indices, self.style_indices).to(self.device)

		self.preprocess = transforms.Compose([
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		self.postprocess = transforms.Compose([
			transforms.Lambda(lambda t: t * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(t.device)),
			transforms.Lambda(lambda t: t + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(t.device)),
		])

	def load_image(self, path: str) -> torch.Tensor:
		image = Image.open(path).convert("RGB")
		return self.preprocess(image).unsqueeze(0).to(self.device)

	def save_image(self, tensor: torch.Tensor, path: str) -> None:
		img = tensor.detach().cpu().clamp(0, 1)[0]
		img = transforms.ToPILImage()(img)
		img.save(path)

	def _compute_losses(
		self,
		gen_c_feats: dict,
		gen_s_feats: dict,
		content_targets: dict,
		style_targets: dict,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		content_loss = 0.0
		for idx in self.content_indices:
			content_loss = content_loss + nn.functional.mse_loss(gen_c_feats[idx], content_targets[idx])

		style_loss = 0.0
		for idx in self.style_indices:
			g_gram = _gram_matrix(gen_s_feats[idx])
			s_gram = style_targets[idx]
			style_loss = style_loss + nn.functional.mse_loss(g_gram, s_gram)

		def tv(x: torch.Tensor) -> torch.Tensor:
			return (
				torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
				torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
			)

		return content_loss, style_loss, tv

	@torch.no_grad()
	def _compute_targets(self, content_img: torch.Tensor, style_img: torch.Tensor) -> Tuple[dict, dict]:
		c_feats, _ = self.feature_extractor(content_img)
		_, s_feats = self.feature_extractor(style_img)
		style_targets = {idx: _gram_matrix(s_feats[idx]) for idx in self.style_indices}
		return c_feats, style_targets

	def transfer_style(
		self,
		content_path: str,
		style_path: str,
		output_path: str = "generated_images/stylized_output.jpg",
		num_iterations: int = 300,
		learning_rate: float = 0.01,
	) -> str:
		_ensure_dir(os.path.dirname(output_path) or ".")
		content_img = self.load_image(content_path)
		style_img = self.load_image(style_path)

		content_targets, style_targets = self._compute_targets(content_img, style_img)

		generated = content_img.clone().requires_grad_(True)
		optimizer = torch.optim.Adam([generated], lr=learning_rate)

		for _ in range(num_iterations):
			optimizer.zero_grad()
			gen_c_feats, gen_s_feats = self.feature_extractor(generated)
			content_loss, style_loss, tv = self._compute_losses(gen_c_feats, gen_s_feats, content_targets, style_targets)
			loss = (
				self.content_weight * content_loss +
				self.style_weight * style_loss +
				self.tv_weight * tv(generated)
			)
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				generated.clamp_(0, 1)

		self.save_image(generated, output_path)
		return output_path





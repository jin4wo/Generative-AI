import os
import argparse
from datetime import datetime

try:
	from neural_style_transfer import NeuralStyleTransfer, _ensure_dir
except ModuleNotFoundError:
	import sys
	sys.path.append(os.path.dirname(__file__))
	from neural_style_transfer import NeuralStyleTransfer, _ensure_dir


def _ensure_sample_images(base_dir: str) -> tuple[str, str]:
	content = os.path.join(base_dir, "content.jpg")
	style = os.path.join(base_dir, "style.jpg")
	if not (os.path.exists(content) and os.path.exists(style)):
		from PIL import Image, ImageDraw
		c = Image.new("RGB", (512, 512), (180, 210, 240))
		s = Image.new("RGB", (512, 512), (240, 180, 210))
		dc = ImageDraw.Draw(c)
		ds = ImageDraw.Draw(s)
		dc.rectangle([64, 64, 448, 448], outline=(20, 60, 120), width=12)
		ds.ellipse([128, 128, 384, 384], outline=(120, 20, 60), width=12)
		c.save(content)
		s.save(style)
	return content, style


def main() -> None:
	parser = argparse.ArgumentParser(description="Quick Neural Style Transfer runner")
	parser.add_argument("--content", type=str, default=None)
	parser.add_argument("--style", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument("--size", type=int, default=192)
	parser.add_argument("--iterations", type=int, default=15)
	parser.add_argument("--lr", type=float, default=0.07)
	parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
	args = parser.parse_args()

	base = os.path.dirname(__file__)
	content_path = args.content or _ensure_sample_images(base)[0]
	style_path = args.style or _ensure_sample_images(base)[1]
	output_path = args.output or os.path.join(base, "output.jpg")

	nst = NeuralStyleTransfer(device=args.device, image_size=args.size)
	_ensure_dir(os.path.dirname(output_path) or ".")

	try:
		result = nst.transfer_style(
			content_path=content_path,
			style_path=style_path,
			output_path=output_path,
			num_iterations=args.iterations,
			learning_rate=args.lr,
		)
		print(f"Saved -> {result}")
	except KeyboardInterrupt:
		print("Interrupted. Partial result (if any) saved at:", output_path)


if __name__ == "__main__":
	main()



import argparse
import os
from shutil import copyfile


def main() -> None:
	parser = argparse.ArgumentParser(description="Create NST pair (content.jpg, style.jpg) from a single input image")
	parser.add_argument("--input", required=True, help="Path to source image (jpg/png)")
	args = parser.parse_args()

	base = os.path.dirname(__file__)
	content_out = os.path.join(base, "content.jpg")
	style_out = os.path.join(base, "style.jpg")

	if not os.path.exists(args.input):
		raise SystemExit(f"Input not found: {args.input}")

	copyfile(args.input, content_out)
	copyfile(args.input, style_out)
	print(f"Wrote -> {content_out}")
	print(f"Wrote -> {style_out}")


if __name__ == "__main__":
	main()





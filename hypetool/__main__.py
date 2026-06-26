# src/hypetool/__main__.py
# That lets you run: python -m hypetool --yaml examples/basic_case/input/inputs.yaml --figures
from hypetool.cli.main import main
if __name__ == "__main__":
    raise SystemExit(main())

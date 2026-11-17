"""Export a torch `team_net` checkpoint to the JSON weight file numpy inference loads.

    python -m aaaag.ml.export --weights final.pt --out model.json [--encoding base64]

Each parameter is a hex string of little-endian float32 bytes (format 1), or with
`--encoding base64` the same bytes base64-encoded (format 2, about two thirds the size: a
d_model 128 network then stays under the repository's 5 MB file limit). The file also carries the
architecture name, the config, the token spec and the feature version, which is everything
`np_infer.load_numpy_weights` needs to rebuild the module. Loading those float32 values is
exact. A config with `switch_head` includes that linear layer; without the flag the parameter
list is the one written before the head existed. A config with `touch_head` includes that linear
layer the same way, after the switch layer when both are on; without the flag the list is
unchanged. This module imports torch; the numpy runner does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_payload
from .np_infer import dump_numpy_weights


def export_file(weights: str | Path, out: str | Path, encoding: str = "hex") -> Path:
    """Write `out` from a `.pt` checkpoint. Returns the path written."""
    payload = load_payload(weights)
    arrays = {
        name: tensor.detach().cpu().float().numpy()
        for name, tensor in payload["state_dict"].items()
    }
    return dump_numpy_weights(
        out,
        arch=payload.get("arch", "team_net"),
        config=payload["model_config"],
        token_spec=payload["token_spec"],
        arrays=arrays,
        feature_version=payload["feature_version"],
        encoding=encoding,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m aaaag.ml.export", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--weights", required=True, help="torch checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="JSON weight file to write")
    parser.add_argument(
        "--encoding", default="hex", choices=("hex", "base64"), help="parameter encoding"
    )
    args = parser.parse_args(argv)
    path = export_file(args.weights, args.out, args.encoding)
    print(f"wrote {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

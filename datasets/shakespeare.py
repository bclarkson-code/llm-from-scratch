from pathlib import Path

import httpx

from tricycle.tokeniser import BPETokeniser


class Shakespeare:
    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    token_path: Path
    raw_data_path: Path
    tokens: list[int]

    def __init__(
        self,
        vocab_size: int,
        token_path: Path = Path("tokens.bin"),
        raw_data_path: Path = Path("raw_data.txt"),
    ):
        self.vocab_size = vocab_size
        self.raw_data_path = raw_data_path
        self.token_path = token_path

        if not self.token_path.exists():
            self.tokens = self.generate()
            self.token_path.write_bytes(bytes(self.tokens))
        else:
            self.tokens = list(self.token_path.read_bytes())

    def download(self):
        raw_data = httpx.get(self.url).text
        with open(self.raw_data_path, "wb") as f:
            f.write(raw_data.encode("utf-8"))

    def generate(self):
        self.download()
        raw_data = list(self.raw_data_path.read_bytes())
        self.tokeniser = BPETokeniser(self.vocab_size)
        self.tokeniser.train_ints(raw_data)
        breakpoint()
        return self.tokeniser.tokenise_ints(raw_data)


if __name__ == "__main__":
    shakespeare = Shakespeare(1024)

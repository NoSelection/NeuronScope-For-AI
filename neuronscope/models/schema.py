from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Metadata about a loaded model."""

    name: str
    path: str
    architecture: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    vocab_size: int
    dtype: str
    device: str
    sliding_window: int | None = None
    has_vision: bool = False
    module_names: list[str] = []

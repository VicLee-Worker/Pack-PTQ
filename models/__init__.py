# Deprecated: transformer-specific quantization helpers moved to utils.initialize
# Kept for backward compatibility if some code imports from models

from utils.initialize import (
    create_base_model,
    modify_attention_layers,
)

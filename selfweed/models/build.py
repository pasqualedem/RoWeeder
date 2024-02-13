from selfweed.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerEncoder


def build_rowweeder_model(
    encoder,
    input_channels,
    embedding_dim=64,
    transformer_layers=4
):
    
    return RowWeeder(
        encoder,
        input_channels=input_channels,
        embedding_dim=embedding_dim,
        transformer_layers=transformer_layers,
    )
    

def build_rowwweder_segformer(
    input_channels,
    embedding_dim=64,
    transformer_layers=4
):
    encoder = SegformerEncoder(
        num_channels=input_channels,
        embed_dim=embedding_dim,
        depth=transformer_layers,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    )
    return build_rowweeder_model(encoder, input_channels, embedding_dim, transformer_layers)
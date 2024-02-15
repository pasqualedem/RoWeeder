from selfweed.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerForImageClassification


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
    

def build_roweeder_segformer(
    input_channels,
    embedding_dim=64,
    transformer_layers=4
):
    encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    encoder = encoder.segformer.encoder
    return build_rowweeder_model(encoder, input_channels, embedding_dim, transformer_layers)
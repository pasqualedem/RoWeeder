from selfweed.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerForImageClassification, SegformerConfig


def build_rowweeder_model(
    encoder,
    input_channels,
    embedding_dims,
    transformer_layers=4
):
    
    return RowWeeder(
        encoder,
        input_channels=input_channels,
        embedding_dims=embedding_dims,
        transformer_layers=transformer_layers,
    )
    

def build_roweeder_segformer(
    input_channels,
    transformer_layers=4
):
    encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    embeddings_dims = SegformerConfig.from_pretrained("nvidia/mit-b0").hidden_sizes
    encoder = encoder.segformer.encoder
    return build_rowweeder_model(encoder, input_channels, embeddings_dims, transformer_layers)
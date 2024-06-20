from selfweed.data.weedmap import WeedMapDataset
from selfweed.models.pseudo import PseudoModel
from selfweed.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerForImageClassification, SegformerConfig, SegformerForSemanticSegmentation

from selfweed.models.utils import HuggingFaceWrapper
from selfweed.data.weedmap import WeedMapDataset

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

def build_segformer(
    input_channels
):
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        id2label=WeedMapDataset.id2class,
        label2id={v: k for k,v in WeedMapDataset.id2class.items()}
        )
    return HuggingFaceWrapper(segformer)


def build_pseudo_gt_model(
    gt_folder
):
    return PseudoModel(gt_folder)

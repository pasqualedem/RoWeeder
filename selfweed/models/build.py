from selfweed.detector import HoughCropRowDetector, get_vegetation_detector
from selfweed.models.pseudo import PseudoModel
from selfweed.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerForImageClassification, SegformerConfig, SegformerForSemanticSegmentation
from transformers import ResNetForImageClassification

from selfweed.models.segmentation import HoughSLICSegmentationWrapper
from selfweed.models.utils import HuggingFaceClassificationWrapper, HuggingFaceWrapper
from selfweed.models.pyramid import MLFormer, PyramidFormer
from selfweed.data.weedmap import WeedMapDataset, ClassificationWeedMapDataset

def build_rowweeder_model(
    encoder,
    input_channels,
    embedding_size,
    embedding_dims,
    transformer_layers=4
):
    
    return RowWeeder(
        encoder,
        input_channels=input_channels,
        embedding_size=embedding_size,
        embedding_dims=embedding_dims,
        transformer_layers=transformer_layers,
    )
    

def build_roweeder_segformer(
    input_channels,
    transformer_layers=4,
    embedding_size=(1, ),
    version="nvidia/mit-b0"
):
    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    encoder = encoder.segformer.encoder
    return build_rowweeder_model(encoder, input_channels, embedding_size, embeddings_dims, transformer_layers)


def build_pyramidformer(
    input_channels,
    version="nvidia/mit-b0",
    fusion="concat",
    upsampling="interpolate"
):
    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    num_classes = len(WeedMapDataset.id2class)
    return PyramidFormer(encoder, embeddings_dims, num_classes, fusion=fusion, upsampling=upsampling)


def build_mlformer(
    input_channels,
    version="nvidia/mit-b0",
    fusion="concat",
    upsampling="interpolate"
):
    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    num_classes = len(WeedMapDataset.id2class)
    return MLFormer(encoder, embeddings_dims, num_classes, fusion=fusion, upsampling=upsampling)


def build_segformer(
    input_channels,
    version="nvidia/mit-b0"
):
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        version,
        id2label=WeedMapDataset.id2class,
        label2id={v: k for k,v in WeedMapDataset.id2class.items()}
        )
    return HuggingFaceWrapper(segformer)


def build_resnet50(
    input_channels,
    plant_detector_params,
    slic_params,
):
    classification_model = HuggingFaceClassificationWrapper(ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        id2label=ClassificationWeedMapDataset.id2class,
        label2id={v: k for k,v in ClassificationWeedMapDataset.id2class.items()},
        ignore_mismatched_sizes=True,
        ))
    plant_detector = get_vegetation_detector(
        plant_detector_params["name"], plant_detector_params["params"]
    )
    return HoughSLICSegmentationWrapper(classification_model, plant_detector, slic_params)


def build_pseudo_gt_model(
    gt_folder
):
    return PseudoModel(gt_folder)

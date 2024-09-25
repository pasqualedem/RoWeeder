from roweeder.detector import HoughCropRowDetector, get_vegetation_detector
from roweeder.models.pseudo import PseudoModel
from roweeder.models.rowweeder import RowWeeder

from transformers.models.segformer.modeling_segformer import SegformerForImageClassification, SegformerConfig, SegformerForSemanticSegmentation
from transformers import ResNetForImageClassification, SwinModel, SwinConfig

from roweeder.models.segmentation import HoughCC, HoughSLIC, HoughSLICSegmentationWrapper
from roweeder.models.utils import HuggingFaceClassificationWrapper, HuggingFaceWrapper, load_state_dict
from roweeder.models.pyramid import RoWeederFlat, RoWeederPyramid
from roweeder.models.utils import torch_dict_load
from roweeder.data.weedmap import WeedMapDataset, ClassificationWeedMapDataset

def build_rowweeder_contrastive(
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
    return build_rowweeder_contrastive(encoder, input_channels, embedding_size, embeddings_dims, transformer_layers)


def build_roweeder_pyramid(
    input_channels,
    version="nvidia/mit-b0",
    fusion="concat",
    upsampling="interpolate",
    blocks=4,
    spatial_conv=True,
    checkpoint=None
):
    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    embeddings_dims = embeddings_dims[:blocks]
    num_classes = len(WeedMapDataset.id2class)
    model = RoWeederPyramid(encoder, num_classes, embeddings_dims, fusion=fusion, upsampling=upsampling, blocks=blocks, spatial_conv=spatial_conv)
    if checkpoint is not None:
        chkpt = torch_dict_load(checkpoint)
        load_state_dict(model, chkpt)
    return model


def build_roweeder_flat(
    input_channels,
    version="nvidia/mit-b0",
    fusion="concat",
    upsampling="interpolate",
    spatial_conv=True,
    blocks=4,
    checkpoint=None,
):
    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    embeddings_dims = embeddings_dims[:blocks]
    num_classes = len(WeedMapDataset.id2class)
    model = RoWeederFlat(encoder, num_classes, embeddings_dims, fusion=fusion, upsampling=upsampling, spatial_conv=spatial_conv)
    if checkpoint is not None:
        chkpt = torch_dict_load(checkpoint)
        load_state_dict(model, chkpt)
    return model


def build_segformer(
    input_channels,
    version="nvidia/mit-b0",
    checkpoint=None
):
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        version,
        id2label=WeedMapDataset.id2class,
        label2id={v: k for k,v in WeedMapDataset.id2class.items()}
        )
    model = HuggingFaceWrapper(segformer)
    if checkpoint is not None:
        chkpt = torch_dict_load(checkpoint)
        load_state_dict(model, chkpt)
    return model


def build_swinmlformer(
    input_channels,
    version="microsoft/swin-tiny-patch4-window7-224",
    fusion="concat",
    upsampling="interpolate",
    spatial_conv=True,
    blocks=5,
    checkpoint=None
):
    encoder = SwinModel.from_pretrained(version)
    config = SwinConfig.from_pretrained(version)
    emb_range = min(blocks, 4)
    embeddings_dims = [config.embed_dim * (2**i) for i in range(emb_range)]
    if blocks > 4:
        embeddings_dims += embeddings_dims[-1:]
    embeddings_dims = embeddings_dims[:blocks]
    scale_factors = [2, 4, 8, 8][:blocks]
    num_classes = len(WeedMapDataset.id2class)
    model = RoWeederFlat(encoder, embeddings_dims, num_classes, fusion=fusion, upsampling=upsampling, spatial_conv=spatial_conv, scale_factors=scale_factors)
    if checkpoint is not None:
        chkpt = torch_dict_load(checkpoint)
        load_state_dict(model, chkpt)
    return model


def build_resnet50(
    input_channels,
    plant_detector_params,
    slic_params,
    internal_batch_size=1,
    checkpoint=None
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
    model = HoughSLICSegmentationWrapper(classification_model, plant_detector, slic_params, internal_batch_size=internal_batch_size)
    if checkpoint is not None:
        chkpt = torch_dict_load(checkpoint)
        load_state_dict(model, chkpt)
    return model


def build_houghcc(
    input_channels,
    plant_detector_params,
    hough_detector_params,
):
    plant_detector = get_vegetation_detector(
        plant_detector_params["name"], plant_detector_params["params"]
    )
    hough_detector = HoughCropRowDetector(**hough_detector_params)
    return HoughCC(plant_detector=plant_detector, hough_detector=hough_detector)


def build_houghslic(
    input_channels,
    plant_detector_params,
    hough_detector_params,
    slic_params,
):
    plant_detector = get_vegetation_detector(
        plant_detector_params["name"], plant_detector_params["params"]
    )
    hough_detector = HoughCropRowDetector(**hough_detector_params)
    return HoughSLIC(plant_detector=plant_detector, hough_detector=hough_detector, slic_params=slic_params)

def build_pseudo_gt_model(
    gt_folder
):
    return PseudoModel(gt_folder)

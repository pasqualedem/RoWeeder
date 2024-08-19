from roweeder.models.build import build_houghcc, build_houghslic, build_roweeder_flat, build_pseudo_gt_model, build_roweeder_pyramid, build_roweeder_segformer, build_segformer, build_resnet50, build_swinmlformer
from roweeder.models.pyramid import RoWeederPyramid, RoWeederFlat


MODEL_REGISTRY = {
    "segformer": build_segformer,
    "rw_segformer": build_roweeder_segformer,
    "pseudo_gt": build_pseudo_gt_model,
    "seg-resnet50": build_resnet50,
    
    "pyramid": build_roweeder_pyramid,
    "pyramidformer": build_roweeder_pyramid,
    
    "flat": build_roweeder_flat,
    "flatformer": build_roweeder_flat,
    "mlformer": build_roweeder_flat,
    
    "swinmlformer": build_swinmlformer,
    "houghcc": build_houghcc,
    "houghslic": build_houghslic,
}


def build_model(params):
    return MODEL_REGISTRY[params["name"]](**params['params'])
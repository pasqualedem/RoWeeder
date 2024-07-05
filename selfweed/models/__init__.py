from selfweed.models.build import build_mlformer, build_pseudo_gt_model, build_pyramidformer, build_roweeder_segformer, build_segformer, build_resnet50


MODEL_REGISTRY = {
    "segformer": build_segformer,
    "rw_segformer": build_roweeder_segformer,
    "pseudo_gt": build_pseudo_gt_model,
    "seg-resnet50": build_resnet50,
    "pyramidformer": build_pyramidformer,
    "mlformer": build_mlformer,
}


def build_model(params):
    return MODEL_REGISTRY[params["name"]](**params['params'])
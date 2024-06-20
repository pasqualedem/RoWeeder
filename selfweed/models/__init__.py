from selfweed.models.build import build_pseudo_gt_model, build_roweeder_segformer, build_segformer


MODEL_REGISTRY = {
    "segformer": build_segformer,
    "rw_segformer": build_roweeder_segformer,
    "pseudo_gt": build_pseudo_gt_model,
}


def build_model(params):
    return MODEL_REGISTRY[params["name"]](**params['params'])
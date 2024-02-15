from selfweed.models.build import build_roweeder_segformer


MODEL_REGISTRY = {
    "rw_segformer": build_roweeder_segformer,
}
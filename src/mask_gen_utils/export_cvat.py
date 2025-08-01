import logging
import fiftyone.utils.cvat as fouc

import hydra
from omegaconf import DictConfig

from src.utils.utils import read_yaml
from src.mask_gen_utils.upload_cvat import MaskRelabelPipeline

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    task_ids = cfg.mask_gen.export_cvat.task_ids
    assert task_ids, "Task IDs must be specified in the config for exporting to CVAT. Check cfg.mask_gen.export_cvat.task_ids"
    keys = read_yaml(cfg.paths.keys_path)

    pipeline = MaskRelabelPipeline(cfg)
    
    samples = pipeline.load_samples_for_relabel()
    dataset = pipeline.inspector.create_dataset(samples)
    view = dataset
        
    fouc.import_annotations(
            view,
            download_media=False,
            task_ids=task_ids,
            username=keys['cvat']['username'],
            password=keys['cvat']['password'],
        )
    
    pipeline.postprocess_annotations(view)
    pipeline.finalize(view, save_to_local=True)

if __name__ == "__main__":
    main()

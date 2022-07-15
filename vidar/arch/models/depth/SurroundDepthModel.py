from vidar.arch.models.BaseModel import BaseModel
from vidar.utils.depth import inv2depth

class SurroundDepthModel(BaseModel):
    """
    PanoDepth model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gt_pose = True
        self._input_keys = ('rgb')

    def forward(self, batch, return_logs=False, **kwargs):
        new_batch = {}
        new_batch['rgb'] = batch['rgb'][0]

        ### Compute depth
        # 1. Compute inverse depth
        net_output = self.networks['depth'](new_batch)

        # TODO(soonminh): remove this and plot validation losses
        # if not return_logs and not self.training:
        if not self.training:
            return {
                'predictions': {
                    'depth': {0: inv2depth(net_output['inv_depths'])},
                },
            }

        raise NotImplementedError

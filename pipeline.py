import kfp
from kfp import dsl
from kfp import onprem
def preprocess_op(pvc_name, volume_name, volume_mount_path):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='white091612/kfp-skin-preprocess:0.1',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 226],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))
def hyp_op(pvc_name, volume_name, volume_mount_path, device):
    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='white091612/kfp-skin-hyp-wandb:1.0',
        arguments=['--data-path', volume_mount_path,
                   '--device', device],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_memory_request('30G').set_memory_limit('40G').set_cpu_request('8').set_cpu_limit('10')
def train_op(pvc_name, volume_name, volume_mount_path, device):
    return dsl.ContainerOp(
        name='Train Model',
        image='white091612/kfp-skin-train:0.7.6',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-name', 'skin-EffNetV2-L',
                   '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_memory_request('30G').set_memory_limit('40G').set_cpu_request('8').set_cpu_limit('10')
    #.apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)
def test_op(pvc_name, volume_name, volume_mount_path, model_path, device):
    return dsl.ContainerOp(
        name='Test Model',
        image='white091612/kfp-skin-test:0.5.2',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-path', model_path,
                   '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))
    #.apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)
@dsl.pipeline(
    name='Skin Cancer Pipeline',
    description=''
)
def surface_pipeline(mode_hyp_train_test: str,
                     preprocess_yes_no: str,
                     model_path: str,
                     device: str):
    pvc_name = "workspace-workspace-skin"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jovyan'
    with dsl.Condition(preprocess_yes_no == 'yes'):
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)
    with dsl.Condition(mode_hyp_train_test == 'hyp'):
        _hyp_op = hyp_op(pvc_name, volume_name, volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'train'):
        _train_op = train_op(pvc_name, volume_name, volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'test'):
        _train_op = test_op(pvc_name, volume_name, volume_mount_path, model_path, device).after(_preprocess_op)
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './skin9_3_eff.yaml')

# from version 9, number of class is 7. before that, it's 2
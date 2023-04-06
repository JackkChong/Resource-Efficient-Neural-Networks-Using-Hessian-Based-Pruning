import torch.nn as nn
from collections import OrderedDict


# =====================================================
# For layer and block dependencies
# =====================================================
def get_layer_dependencies(model, network):
    """
    There should be a total of X dependencies in the dictionary

    1 empty dependency for the 1st conv layer
    1 dependency for the last fc layer
    """

    dependencies = OrderedDict()

    # if network == "wideresnet":
    #     dependencies[model.conv1] = []
    #     prev_modules = [model.conv1]

    #     update_wideresnet_layer_dependencies(prev_modules, model.layer1, dependencies)
    #     prev_modules = [model.layer1[-1].conv2]
    #     if model.layer1[-1].convShortcut is not None:
    #         prev_modules.append(model.layer1[-1].convShortCut)
    #     else:
    #         prev_modules = [model.layer1[-1].conv2] + dependencies[model.layer1[-1].conv1]

    #     update_wideresnet_layer_dependencies(prev_modules, model.layer2, dependencies)
    #     prev_modules = [model.layer2[-1].conv2]
    #     if model.layer2[-1].convShortcut is not None:
    #         prev_modules.append(model.layer2[-1].convShortcut)
    #     else:
    #         prev_modules = [model.layer2[-1].conv2] + dependencies[model.layer2[-1].conv1]

    #     update_wideresnet_layer_dependencies(prev_modules, model.layer3, dependencies)
    #     prev_modules = [model.layer3[-1].conv2]
    #     if model.layer3[-1].convShortcut is not None:
    #         prev_modules.append(model.layer3[-1].convShortCut)
    #     else:
    #         prev_modules = [model.layer3[-1].conv2] + dependencies[model.layer3[-1].conv1]

    #     dependencies[model.bn1] = prev_modules
    #     dependencies[model.fc] = prev_modules

    if network == "resnet" or network == "wideresnet":
        dependencies[model.features.init_block.conv] = []
        dependencies[model.features.init_block.bn] = [model.features.init_block.conv]

        prev_modules = [model.features.init_block.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage1, dependencies)

        prev_modules = [model.features.stage1[-1].body.conv2.conv]
        if hasattr(model.features.stage1[-1], "identity_conv"):
            prev_modules.append(model.features.stage1[-1].identity_conv)
        else:
            prev_modules = [model.features.stage1[-1].body.conv2.conv] + dependencies[
                model.features.stage1[-1].body.conv1.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage2, dependencies)

        prev_modules = [model.features.stage2[-1].body.conv2.conv]
        if hasattr(model.features.stage2[-1], "identity_conv"):
            prev_modules.append(model.features.stage2[-1].identity_conv)
        else:
            prev_modules = [model.features.stage2[-1].body.conv2.conv] + dependencies[
                model.features.stage2[-1].body.conv1.conv]
        update_resnet_stage_dependencies(prev_modules, model.features.stage3, dependencies)

        prev_modules = [model.features.stage3[-1].body.conv2.conv]
        if hasattr(model.features.stage3[-1], "identity_conv"):
            prev_modules.append(model.features.stage3[-1].identity_conv)
        else:
            prev_modules = [model.features.stage3[-1].body.conv2.conv] + dependencies[
                model.features.stage3[-1].body.conv1.conv]

        dependencies[model.output] = prev_modules

    return dependencies


def update_resnet_stage_dependencies(prev_modules, stage, dependencies):
    num_units = len(stage)
    for unit_idx in range(num_units):
        unit = stage[unit_idx]
        update_resnet_unit_dependencies(prev_modules, unit, dependencies)
        prev_modules = [unit.body.conv2.conv]
        if hasattr(unit, "identity_conv"):
            prev_modules.append(unit.identity_conv.conv)
        else:
            prev_modules.extend(dependencies[unit.body.conv1.conv])


def update_resnet_unit_dependencies(prev_modules, unit, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'

    dependencies[unit.body.conv1.conv] = prev_modules
    dependencies[unit.body.conv1.bn] = [unit.body.conv1.conv]
    dependencies[unit.body.conv2.conv] = [unit.body.conv1.conv]
    dependencies[unit.body.conv2.bn] = [unit.body.conv2.conv]

    if hasattr(unit, "identity_conv"):
        dependencies[unit.identity_conv.conv] = prev_modules
        dependencies[unit.identity_conv.bn] = [unit.identity_conv.conv]

import omni.client
import omni.usd
from pxr import Sdf, UsdGeom

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="convert an asset to an instanceable asset."
)
parser.add_argument(
    "--source_usd", type=str, default=None, help="path to the source usd"
)
parser.add_argument(
    "--target_usd", type=str, default=None, help="path to the target usd"
)
parser.add_argument("--prim_path", type=str, default=None, help="prim path")

args_cli = parser.parse_args()


def update_reference(source_prim_path, source_reference_path, target_reference_path):
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    while len(prims) > 0:
        prim = prims.pop(0)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
        reference_list = prim_spec.referenceList
        refs = reference_list.GetAddedOrExplicitItems()
        if len(refs) > 0:
            for ref in refs:
                if ref.assetPath == source_reference_path:
                    prim.GetReferences().RemoveReference(ref)
                    prim.GetReferences().AddReference(
                        assetPath=target_reference_path, primPath=prim.GetPath()
                    )

        prims = prims + prim.GetChildren()


def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
    Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(
                Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0)
            )
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)


def convert_asset_instanceable(
    asset_usd_path, source_prim_path, save_as_path=None, create_xforms=True
):
    """Makes all mesh/geometry prims instanceable.
    Can optionally add UsdGeom.Xform prim as parent for all mesh/geometry prims.
    Makes a copy of the asset USD file, which will be used for referencing.
    Updates asset file to convert all parent prims of mesh/geometry prims to reference cloned USD file.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
        create_xforms (bool): Whether to add new UsdGeom.Xform prims to mesh/geometry prims.
    """

    if create_xforms:
        create_parent_xforms(asset_usd_path, source_prim_path, save_as_path)
        asset_usd_path = save_as_path

    instance_usd_path = ".".join(asset_usd_path.split(".")[:-1]) + "_meshes.usd"
    omni.client.copy(asset_usd_path, instance_usd_path)
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    while len(prims) > 0:
        prim = prims.pop(0)
        if prim:
            if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                parent_prim = prim.GetParent()
                if parent_prim and not parent_prim.IsInstance():
                    parent_prim.GetReferences().AddReference(
                        assetPath=instance_usd_path, primPath=str(parent_prim.GetPath())
                    )
                    parent_prim.SetInstanceable(True)
                    continue

            children_prims = prim.GetChildren()
            prims = prims + children_prims

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)


ASSET_USD_PATH = "/home/liusong/git/isacc_lab_galaxea/source/extensions/omni.isaac.lab_assets/data/Robots/Galaxea/r1_DVT_colored.usd"
SOURCE_PRIM_PATH = "/R1"
SAVE_AS_PATH = "/home/liusong/git/isacc_lab_galaxea/source/extensions/omni.isaac.lab_assets/data/Robots/Galaxea/r1_DVT_colored_instancable.usd"

convert_asset_instanceable(ASSET_USD_PATH, SOURCE_PRIM_PATH, SAVE_AS_PATH)
# convert_asset_instanceable(args_cli.source_usd, args_cli.prim_path, args_cli.target_usd)

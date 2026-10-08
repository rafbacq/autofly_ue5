"""UE-side helpers shared by the import script and the material fix-up (UE embedded Python; stdlib + unreal only).

Why foliage must be Masked: Interchange imports a glTF material of alphaMode BLEND (Poly Haven's leaves and twigs) as a
Translucent instance of its M_Default master. Translucent surfaces write no depth, so the drone's depth camera -- the
only thing the depth-only expert flies on -- saw trunks and bare branches where the RGB showed full canopies, while the
collision mesh still held every leaf card (s02's first live check, 2026-10-07). UE foliage is alpha-tested (Masked):
the same instance with its blend mode overridden to Masked renders the cards' opaque texels into depth and discards the
transparent ones. The glTF alphaCutoff (0.5 when unspecified) is what the instance's AlphaCutoff parameter and the
opacity-mask clip carry.
"""
import unreal

MEL = unreal.MaterialEditingLibrary
DEFAULT_ALPHA_CUTOFF = 0.5  # glTF 2.0's default alphaCutoff for MASK materials


def enum_name(value):
    return str(value).split(".")[-1].split(":")[0]


def materials_of(mesh):
    out = []
    for slot in mesh.get_editor_property("static_materials"):
        mi = slot.get_editor_property("material_interface")
        if mi is not None:
            out.append(mi)
    return out


def mask_translucent(material, alpha_cutoff=DEFAULT_ALPHA_CUTOFF):
    """Override a translucent material instance to Masked with `alpha_cutoff`; returns what was done, or None when the
    material was not a translucent instance. Saves the instance."""
    if not isinstance(material, unreal.MaterialInstanceConstant):
        return None
    if enum_name(material.get_blend_mode()) != "BLEND_TRANSLUCENT":
        return None
    overrides = material.get_editor_property("base_property_overrides")
    overrides.set_editor_property("override_blend_mode", True)
    overrides.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    overrides.set_editor_property("override_opacity_mask_clip_value", True)
    overrides.set_editor_property("opacity_mask_clip_value", float(alpha_cutoff))
    material.set_editor_property("base_property_overrides", overrides)
    if "AlphaCutoff" in [str(n) for n in MEL.get_scalar_parameter_names(material)]:
        MEL.set_material_instance_scalar_parameter_value(material, "AlphaCutoff", float(alpha_cutoff))
    MEL.update_material_instance(material)
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, False):
        raise RuntimeError("could not save " + material.get_path_name())
    return {"material": material.get_path_name(), "blend_mode": enum_name(material.get_blend_mode()), "alpha_cutoff": float(alpha_cutoff)}


def mask_foliage_of(mesh, alpha_cutoff=DEFAULT_ALPHA_CUTOFF):
    """Every translucent material slot of a static mesh made Masked; returns the list of changes and the blend mode of
    every slot afterwards, for the report."""
    changed = []
    for material in materials_of(mesh):
        done = mask_translucent(material, alpha_cutoff)
        if done is not None:
            changed.append(done)
    return {"changed": changed,
            "slots": [{"material": m.get_path_name(), "blend_mode": enum_name(m.get_blend_mode())} for m in materials_of(mesh)]}

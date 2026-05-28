# mjlab_cfgs/assets/zoo/robots/point_mass/point_mass_constants.py
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg

_CURRENT_DIR = Path(__file__).resolve().parent
POINT_MASS_XML = _CURRENT_DIR / "xmls" / "point_mass.xml"

def get_spec() -> mujoco.MjSpec:
    if not POINT_MASS_XML.exists():
        raise FileNotFoundError(f"XML not found: {POINT_MASS_XML}")
    return mujoco.MjSpec.from_file(str(POINT_MASS_XML))

def get_point_mass_robot_cfg() -> EntityCfg:
    """Point mass: 2 DOF (x, y), force control."""
    actuators = (XmlActuatorCfg(target_names_expr=("free_x", "free_y")),)
    articulation = EntityArticulationInfoCfg(actuators=actuators)
    return EntityCfg(spec_fn=get_spec, articulation=articulation)
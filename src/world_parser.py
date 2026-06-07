"""Parse the Gazebo world SDF to extract crop positions and colors."""
import re
from dataclasses import dataclass
from pathlib import Path

from src.disease_constants import color_to_disease

WORLD_PATH = Path("/home/aditya/agribot_ws/world/agribot_farm_world.sdf")


@dataclass(frozen=True)
class InfectedCrop:
    id: int
    gz_x: float
    gz_y: float
    gz_z: float
    disease_type: str = "Infected"


@dataclass(frozen=True)
class SphericalOrigin:
    lat: float
    lon: float
    alt: float


def parse_infected_crops(world_path: Path) -> list[InfectedCrop]:
    """Return the diseased crop positions from the world file."""
    text = Path(world_path).read_text()
    pattern = re.compile(
        r'<model name="crop_(\d+)">(.*?)</model>',
        re.DOTALL,
    )
    crops: list[InfectedCrop] = []
    for match in pattern.finditer(text):
        crop_id = int(match.group(1))
        body = match.group(2)
        ambient_m = re.search(
            r'<ambient>([\d.]+) ([\d.]+) ([\d.]+)', body
        )
        if ambient_m is None:
            continue
        r, g, b = (float(ambient_m.group(i)) for i in (1, 2, 3))
        disease = color_to_disease(r, g, b)
        if disease == "Healthy":
            continue
        pose_m = re.search(r'<pose>([\d.\-]+) ([\d.\-]+) ([\d.\-]+)', body)
        if pose_m is None:
            continue
        crops.append(InfectedCrop(
            id=crop_id,
            gz_x=float(pose_m.group(1)),
            gz_y=float(pose_m.group(2)),
            gz_z=float(pose_m.group(3)),
            disease_type=disease,
        ))
    return crops


def parse_spherical_origin(world_path: Path) -> SphericalOrigin:
    """Return the world file's <spherical_coordinates> values."""
    text = Path(world_path).read_text()
    match = re.search(
        r"<spherical_coordinates>(.*?)</spherical_coordinates>",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("No <spherical_coordinates> in world file")
    block = match.group(1)
    lat = float(re.search(r"<latitude_deg>([^<]+)</latitude_deg>", block).group(1))
    lon = float(re.search(r"<longitude_deg>([^<]+)</longitude_deg>", block).group(1))
    alt_match = re.search(r"<elevation>([^<]+)</elevation>", block)
    alt = float(alt_match.group(1)) if alt_match else 0.0
    return SphericalOrigin(lat=lat, lon=lon, alt=alt)

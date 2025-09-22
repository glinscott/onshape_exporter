"""Minimal Onshape assembly exporter with optional motion baking.

This module factors out the original tutorial-specific converter into a
reusable command line tool. It can:
  * Download tessellated faces/edges for the supplied assembly
  * Build a GLB scene + occurrence to node map
  * Optionally sample a configuration parameter across angles/values to bake
    motion transforms
  * Optionally run gltfpack to optimise the resulting GLB (if available)

Usage:
  python onshape_exporter.py --assembly <assembly_url> --out out/

Credentials:
  Provide Onshape API credentials via environment variables
    ONSHAPE_ACCESS_KEY
    ONSHAPE_SECRET_KEY
  or pass them explicitly via command line flags.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import trimesh
from trimesh.path import Path3D

# Onshape uses uppercase booleans in JSON payloads. requests handles this, but
# flake8 would normally flag the casing; we leave it as-is for API compatibility.

DEFAULT_BASE_URL = "https://cad.onshape.com"
MATRIX_DECIMALS = 6


def log(message: str):
  print(message, flush=True)


@dataclass
class ElementRef:
  document_id: str
  wvm: str
  wvm_id: str
  element_id: str


class OnshapeClient:
  def __init__(self, access_key: str, secret_key: str, base_url: str = DEFAULT_BASE_URL):
    if not access_key or not secret_key:
      raise ValueError("Onshape API keys are required. Set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY or use CLI flags.")
    self.access_key = access_key
    self.secret_key = secret_key
    self.base_url = base_url.rstrip('/')
    self.headers = {
      "Accept": "application/json;charset=UTF-8;qs=0.09",
      "Content-Type": "application/json",
    }

  def get_json(self, path: str, *, params: Optional[Dict[str, object]] = None) -> dict:
    url = f"{self.base_url}/{path.lstrip('/')}"
    response = requests.get(url, params=params, headers=self.headers, auth=(self.access_key, self.secret_key))
    response.raise_for_status()
    return response.json()

  def post_json(self, path: str, *, body: dict) -> dict:
    url = f"{self.base_url}/{path.lstrip('/')}"
    response = requests.post(url, json=body, headers=self.headers, auth=(self.access_key, self.secret_key))
    response.raise_for_status()
    return response.json()


def parse_element_ref(ref: str) -> ElementRef:
  """Parse an Onshape element reference from a URL or colon-separated string."""
  if "documents" in ref:
    from urllib.parse import urlparse

    parsed = urlparse(ref)
    parts = [p for p in parsed.path.split('/') if p]
    try:
      doc_idx = parts.index("documents")
    except ValueError as exc:
      raise ValueError("Assembly URL must include /documents/<id>/...") from exc
    try:
      document_id = parts[doc_idx + 1]
      wvm = parts[doc_idx + 2]
      if wvm not in {"w", "v", "m"}:
        raise ValueError
      wvm_id = parts[doc_idx + 3]
      if parts[doc_idx + 4] != "e":
        raise ValueError
      element_id = parts[doc_idx + 5]
    except (IndexError, ValueError) as exc:
      raise ValueError("Unable to parse assembly URL; expected /documents/<id>/(w|v|m)/<id>/e/<element>.") from exc
    return ElementRef(document_id, wvm, wvm_id, element_id)

  tokens = [t for t in ref.split(":") if t]
  if len(tokens) != 4:
    raise ValueError("Element reference must be either a full URL or did:wvm:wvmid:eid")
  document_id, wvm, wvm_id, element_id = tokens
  if wvm not in {"w", "v", "m"}:
    raise ValueError("wvm must be one of 'w' (workspace), 'v' (version), or 'm' (microversion)")
  return ElementRef(document_id, wvm, wvm_id, element_id)


def base64_to_linear_rgba(b64str: str) -> np.ndarray:
  rgba_bytes = base64.b64decode(b64str)
  srgb = np.frombuffer(rgba_bytes, dtype=np.uint8) / 255.0
  rgb = srgb[:3]
  linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
  return np.concatenate((linear_rgb, srgb[3:4]))


def _sanitize(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _quantize_float(value: float, decimals: int = MATRIX_DECIMALS) -> float:
  rounded = round(float(value), decimals)
  return 0.0 if rounded == -0.0 else rounded


def _quantize_matrix(mat: np.ndarray) -> Tuple[float, ...]:
  return tuple(_quantize_float(v) for v in mat.flatten())


def assembly_path(ref: ElementRef) -> str:
  return f"api/assemblies/d/{ref.document_id}/{ref.wvm}/{ref.wvm_id}/e/{ref.element_id}/"


def configuration_path(ref: ElementRef) -> str:
  return f"api/v6/elements/d/{ref.document_id}/{ref.wvm}/{ref.wvm_id}/e/{ref.element_id}/configuration"


def configuration_encoding_path(ref: ElementRef) -> str:
  return f"api/v6/elements/d/{ref.document_id}/e/{ref.element_id}/configurationencodings"


def partstudio_faces_path(ref: ElementRef, partstudio_id: str) -> str:
  return f"api/partstudios/d/{ref.document_id}/{ref.wvm}/{ref.wvm_id}/e/{partstudio_id}/tessellatedfaces"


def partstudio_edges_path(ref: ElementRef, partstudio_id: str) -> str:
  return f"api/partstudios/d/{ref.document_id}/{ref.wvm}/{ref.wvm_id}/e/{partstudio_id}/tessellatededges"


def fetch_assembly(client: OnshapeClient, ref: ElementRef, *, configuration: Optional[str] = None) -> dict:
  params = {"configuration": configuration} if configuration else None
  return client.get_json(assembly_path(ref), params=params)


def fetch_partstudio_faces(client: OnshapeClient, ref: ElementRef, partstudio_ids: Iterable[str]) -> List[dict]:
  faces: List[dict] = []
  params = dict(
    rollbackBarIndex=-1,
    outputFaceAppearances=True,
    outputVertexNormals=True,
    outputFacetNormals=False,
    outputTextureCoordinates=False,
    outputIndexTable=False,
    outputErrorFaces=False,
    combineCompositePartConstituents=False,
    chordTolerance=0.0001,
    angleTolerance=1,
  )
  for ps_id in partstudio_ids:
    faces.extend(client.get_json(partstudio_faces_path(ref, ps_id), params=params))
  return faces


def fetch_partstudio_edges(client: OnshapeClient, ref: ElementRef, partstudio_ids: Iterable[str]) -> List[dict]:
  edges: List[dict] = []
  params = dict(rollbackBarIndex=-1, chordTolerance=0.0001, angleTolerance=1)
  for ps_id in partstudio_ids:
    edges.extend(client.get_json(partstudio_edges_path(ref, ps_id), params=params))
  return edges


def discover_partstudio_ids(assembly: dict) -> List[str]:
  ids = sorted({inst["elementId"] for inst in assembly["rootAssembly"].get("instances", []) if "partId" in inst})
  if not ids:
    raise RuntimeError("No Part Studio element IDs discovered. Pass --partstudio explicitly.")
  return ids


def build_scene_and_occmap(assembly: dict, faces: Sequence[dict], edges: Sequence[dict]) -> Tuple[trimesh.Scene, Dict[str, Dict[str, str]]]:
  scene = trimesh.Scene()
  edge_geom: Dict[str, Path3D] = {}

  for part in edges:
    part_id = part.get("id")
    edge_list = [np.asarray(edge["vertices"]) for edge in part.get("edges", [])]
    if not part_id or not edge_list:
      continue
    segments_per_edge = [np.stack((edge[:-1], edge[1:]), axis=1) for edge in edge_list]
    segments = np.concatenate(segments_per_edge, axis=0)
    path = trimesh.load_path(segments)
    black = np.array([0, 0, 0, 255], dtype=np.uint8)
    path.colors = np.tile(black, (len(path.entities), 1))
    edge_geom[f"{part_id}_edges"] = path

  solid_geom: Dict[str, trimesh.Trimesh] = {}
  for body in faces:
    body_id = body.get("id")
    if not body_id:
      continue
    triangle_list = []
    for face in body.get("faces", []):
      facets = face.get("facets", [])
      if facets:
        triangle_list.append(np.asarray([f["vertices"] for f in facets], dtype=np.float32))
    if not triangle_list:
      continue
    triangles = np.concatenate(triangle_list, axis=0)
    vertices = triangles.reshape(-1, 3)
    faces_idx = np.arange(len(vertices)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_idx, process=False)
    if "color" in body:
      mesh.visual.vertex_colors = base64_to_linear_rgba(body["color"])
    solid_geom[body_id] = mesh

  instances = {inst["id"]: inst for inst in assembly["rootAssembly"].get("instances", [])}
  occ2node: Dict[str, Dict[str, str]] = {}
  for idx, occ in enumerate(assembly["rootAssembly"].get("occurrences", [])):
    path_tokens = occ.get("path", [])
    if not path_tokens:
      continue
    leaf_id = path_tokens[-1]
    inst = instances.get(leaf_id)
    if not inst or "partId" not in inst:
      continue
    part_id = inst["partId"]
    mesh = solid_geom.get(part_id)
    if mesh is None:
      continue
    transform = occ.get("transform", [])
    if len(transform) != 16:
      continue
    matrix44 = np.asarray(transform, float).reshape(4, 4)
    solid_node = _sanitize(f"{part_id}_{idx}")
    scene.add_geometry(mesh, transform=matrix44, node_name=solid_node)

    edge_node = None
    edge_key = f"{part_id}_edges"
    if edge_key in edge_geom:
      edge_node = _sanitize(f"{edge_key}_{idx}")
      scene.add_geometry(edge_geom[edge_key], transform=matrix44, node_name=edge_node)

    occ_key = "/".join(path_tokens)
    occ2node[occ_key] = {"solid": solid_node}
    if edge_node:
      occ2node[occ_key]["edges"] = edge_node

  return scene, occ2node


def resolve_configuration_parameter(client: OnshapeClient, ref: ElementRef, parameter_name: str) -> str:
  config = client.get_json(configuration_path(ref))
  for param in config.get("configurationParameters", []):
    if param.get("parameterName") == parameter_name:
      return param["parameterId"]
  raise RuntimeError(f"Configuration parameter '{parameter_name}' not found.")


def encode_configuration_value(client: OnshapeClient, ref: ElementRef, parameter_id: str, value_literal: str) -> str:
  body = {"parameters": [{"parameterId": parameter_id, "parameterValue": value_literal}]}
  response = client.post_json(configuration_encoding_path(ref), body=body)
  encoded = response.get("encodedId")
  if not encoded:
    raise RuntimeError("Onshape did not return an encoded configuration id")
  return encoded


def sample_motion(
  client: OnshapeClient,
  ref: ElementRef,
  parameter_name: str,
  values: Sequence[float],
  value_template: str,
  progress: Optional[Callable[[int, float, int], None]] = None,
) -> dict:
  parameter_id = resolve_configuration_parameter(client, ref, parameter_name)
  frames: List[dict] = []
  matrix_table: List[Tuple[float, ...]] = []
  matrix_lookup: Dict[Tuple[float, ...], int] = {}

  total = len(values)
  for idx, value in enumerate(values):
    if progress:
      progress(idx, value, total)
    literal = value_template.format(value=value)
    encoded_config = encode_configuration_value(client, ref, parameter_id, literal)
    assembly = fetch_assembly(client, ref, configuration=encoded_config)
    frame_occurrences: Dict[str, int] = {}
    for occ in assembly["rootAssembly"].get("occurrences", []):
      key = "/".join(occ.get("path", []))
      transform = occ.get("transform", [])
      if len(transform) != 16:
        continue
      matrix = np.asarray(transform, float).reshape(4, 4)
      quantized = _quantize_matrix(matrix)
      matrix_idx = matrix_lookup.get(quantized)
      if matrix_idx is None:
        matrix_idx = len(matrix_table)
        matrix_lookup[quantized] = matrix_idx
        matrix_table.append(quantized)
      frame_occurrences[key] = matrix_idx
    frames.append({"value": value, "occurrences": frame_occurrences})

  return {
    "metadata": {
      "documentId": ref.document_id,
      "wvm": ref.wvm,
      "wvmId": ref.wvm_id,
      "elementId": ref.element_id,
      "parameter": parameter_name,
      "valueTemplate": value_template,
    },
    "values": list(values),
    "matrixTable": [list(vals) for vals in matrix_table],
    "frames": frames,
  }


def export_glb(scene: trimesh.Scene, out_path: str):
  glb = trimesh.exchange.gltf.export_glb(scene)
  with open(out_path, "wb") as f:
    f.write(glb)


def optimise_glb(src_path: str, dst_path: str, *, gltfpack: str, extra_args: Optional[Sequence[str]] = None):
  cmd = [gltfpack, "-i", src_path, "-o", dst_path]
  if extra_args:
    cmd.extend(extra_args)
  subprocess.run(cmd, check=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export a GLB + motion data from an Onshape assembly.")
  parser.add_argument("--assembly", required=True, help="Onshape assembly element URL or did:wvm:wvmid:eid")
  parser.add_argument("--partstudio", nargs="*", default=None, help="Explicit Part Studio element IDs to tessellate")
  parser.add_argument("--out", default="out", help="Output directory (default: out)")
  parser.add_argument("--faces-cache", default=None, help="Optional cache JSON for faces to avoid refetching")
  parser.add_argument("--edges-cache", default=None, help="Optional cache JSON for edges to avoid refetching")
  parser.add_argument("--skip-edges", action="store_true", help="Skip edge tessellation to speed up export")
  parser.add_argument("--access-key", default=os.environ.get("ONSHAPE_ACCESS_KEY"), help="Onshape access key (or set ONSHAPE_ACCESS_KEY)")
  parser.add_argument("--secret-key", default=os.environ.get("ONSHAPE_SECRET_KEY"), help="Onshape secret key (or set ONSHAPE_SECRET_KEY)")
  parser.add_argument("--base-url", default=os.environ.get("ONSHAPE_BASE_URL", DEFAULT_BASE_URL), help="Onshape API base URL")
  parser.add_argument("--motion-parameter", default=None, help="Configuration parameter name to sweep for motion baking")
  parser.add_argument("--angles", type=float, nargs="*", help="Angles/values to sample for motion (requires --motion-parameter)")
  parser.add_argument("--frame-count", type=int, default=20, help="Number of evenly spaced samples over 0..360 (exclusive) if --angles not supplied")
  parser.add_argument("--value-template", default="{value} degree", help="Literal template for configuration values (default: '{value} degree')")
  parser.add_argument("--optimise", action="store_true", help="Run gltfpack on the exported GLB")
  parser.add_argument("--gltfpack", default=os.environ.get("GLTFPACK_PATH", "gltfpack"), help="Path to gltfpack executable")
  parser.add_argument("--gltfpack-args", nargs="*", default=None, help="Additional arguments to pass to gltfpack")
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = parse_args(argv)
  client = OnshapeClient(args.access_key, args.secret_key, base_url=args.base_url)
  ref = parse_element_ref(args.assembly)

  os.makedirs(args.out, exist_ok=True)

  log(f"Fetching assembly: {args.assembly}")
  assembly0 = fetch_assembly(client, ref)
  log("Assembly fetched.")

  if args.partstudio:
    partstudio_ids = args.partstudio
    log(f"Using provided Part Studio IDs ({len(partstudio_ids)}): {', '.join(partstudio_ids)}")
  else:
    log("Discovering Part Studio IDs from assembly...")
    partstudio_ids = discover_partstudio_ids(assembly0)
    log(f"Found {len(partstudio_ids)} Part Studio IDs.")

  if args.faces_cache and os.path.exists(args.faces_cache):
    log(f"Loading faces from cache {args.faces_cache}")
    with open(args.faces_cache) as f:
      faces = json.load(f)
  else:
    log(f"Fetching tessellated faces for {len(partstudio_ids)} Part Studios...")
    faces = fetch_partstudio_faces(client, ref, partstudio_ids)
    log(f"Fetched {len(faces)} bodies worth of faces.")
    if args.faces_cache:
      log(f"Caching faces to {args.faces_cache}")
      with open(args.faces_cache, "w") as f:
        json.dump(faces, f, indent=2)

  if args.skip_edges:
    log("Skipping edge tessellation (--skip-edges).")
    edges: List[dict] = []
  elif args.edges_cache and os.path.exists(args.edges_cache):
    log(f"Loading edges from cache {args.edges_cache}")
    with open(args.edges_cache) as f:
      edges = json.load(f)
  else:
    log(f"Fetching tessellated edges for {len(partstudio_ids)} Part Studios...")
    edges = fetch_partstudio_edges(client, ref, partstudio_ids)
    log(f"Fetched {len(edges)} edge entries.")
    if args.edges_cache:
      log(f"Caching edges to {args.edges_cache}")
      with open(args.edges_cache, "w") as f:
        json.dump(edges, f, indent=2)

  log("Building scene and occurrence map...")
  scene, occ2node = build_scene_and_occmap(assembly0, faces, edges)

  glb_path = os.path.join(args.out, "assembly.glb")
  export_glb(scene, glb_path)
  log(f"Wrote GLB to {glb_path}")

  occ_path = os.path.join(args.out, "occ2node.json")
  with open(occ_path, "w") as f:
    json.dump(occ2node, f, indent=2)
  log(f"Wrote occurrence map to {occ_path}")

  if args.optimise:
    optimised_path = os.path.join(args.out, "assembly.optimised.glb")
    log("Optimising GLB with gltfpack...")
    optimise_glb(glb_path, optimised_path, gltfpack=args.gltfpack, extra_args=args.gltfpack_args)
    log(f"Optimised GLB written to {optimised_path}")

  if args.motion_parameter:
    if args.angles:
      values = args.angles
    else:
      step = 360.0 / float(args.frame_count)
      values = [i * step for i in range(args.frame_count)]
    log(f"Baking motion for parameter {args.motion_parameter} over {len(values)} samples...")
    def _progress(idx, value, total):
      literal = args.value_template.format(value=value)
      log(f"  Sampling {idx + 1}/{total}: {literal}")
    motion = sample_motion(client, ref, args.motion_parameter, values, args.value_template, progress=_progress)
    motion_path = os.path.join(args.out, "motion.json")
    motion_str = json.dumps(motion, separators=(",", ":"))
    with open(motion_path, "w") as f:
      f.write(motion_str)
    with gzip.open(f"{motion_path}.gz", "wt", encoding="utf-8") as f:
      f.write(motion_str)
    log(f"Motion tables written to {motion_path} (and gzip variant).")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

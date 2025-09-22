# Onshape Exporter

A reusable Python tool that downloads an Onshape assembly, writes a GLB with optional edge overlays, and (optionally) bakes motion transforms by sweeping a configuration parameter. 

<img width="928" height="531" alt="image" src="https://github.com/user-attachments/assets/b0c497ce-c455-4dc8-83f0-e5b6e2776ef8" />

## Features
- Fetch tessellated faces (and optional edges) for any assembly element
- Build a GLB plus an `occ2node.json` map that records which occurrences map to which scene nodes
- Optionally sweep a configuration parameter (for example `thetaDeg`) to bake motion frames and deduplicate transforms
- Optional GLB optimisation via [gltfpack](https://github.com/zeux/meshoptimizer/tree/master/gltf) if the binary is available on your system
- Minimal CLI with sensible defaults, credential discovery, and lightweight caching for geometry JSON dumps

## Installation
You can run the exporter in-place with [uv](https://github.com/astral-sh/uv) or install it as a standalone tool. It depends on numpy, requests, trimesh, and SciPy; both uv and pip will install these automatically.

### Run with uv (recommended during development)
```
uv run exporter/onshape_exporter.py --assembly <assembly_url> --out out
```

### Install as a package
```
uv pip install ./exporter
# or: pip install ./exporter
onshape-exporter --assembly <assembly_url> --out out
```

## Authentication
Set the Onshape API credentials as environment variables or pass them via CLI flags:

```
export ONSHAPE_ACCESS_KEY=...  # required
export ONSHAPE_SECRET_KEY=...  # required
export ONSHAPE_BASE_URL=https://cad.onshape.com  # optional override
```

Alternatively:
```
onshape-exporter --assembly <assembly_url> \
  --access-key $ACCESS_KEY \
  --secret-key $SECRET_KEY
```

## Command reference
```
onshape-exporter --assembly <url_or_ids> [options]
```

Key options:
- `--assembly`: Required. Either a full Onshape URL (`https://cad.onshape.com/documents/<did>/w/<wvmid>/e/<eid>`) or the shorthand `did:w:v_or_m_or_wid:eid`.
- `--partstudio`: Explicit Part Studio element IDs to tessellate. Defaults to auto-discovery from the assembly.
- `--out`: Output directory (default `out`). Files: `assembly.glb`, `occ2node.json`, optional `motion.json(.gz)`, optional `assembly.optimised.glb`.
- `--faces-cache`, `--edges-cache`: Cache JSON files to avoid re-fetching tessellation data.
- `--skip-edges`: Skip tessellated edges if you only need solid geometry.
- `--motion-parameter`: Name of the configuration parameter to sweep (e.g. `thetaDeg`). When supplied, you can pass explicit `--angles <list>` or rely on evenly spaced samples with `--frame-count` (default 20).
- `--value-template`: Template string for the configuration literal. The placeholder `{value}` is replaced by the sampled value (default `{value} degree`).
- `--optimise`: Run gltfpack after export (set `--gltfpack` if it is not on `PATH`). Additional arguments can be passed with `--gltfpack-args`.

## Example
```
uv run onshape_exporter.py \
  --assembly https://cad.onshape.com/documents/ebc9190f428cf30153c06148/w/8e27fa4d26837b5b136fb4a1/e/d2f73f6396ee11d44c08fc80 \
  --motion-parameter thetaDeg \
  --frame-count 36 \
  --optimise \
  --out ./out
```
This writes the GLB, occurrence map, motion table, and an optimised GLB (if gltfpack is present).

## View the output
```
python -m http.server 8000
# open http://localhost:8000/
```
The viewer automatically loads `out/assembly.glb`, `out/occ2node.json`, and (if present) `out/motion.json`.

## License
MIT

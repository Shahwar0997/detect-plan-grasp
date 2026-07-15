#!/usr/bin/env bash
# Restore the Franka Panda mesh assets (gitignored — 33 MB of binaries) from DeepMind's
# mujoco_menagerie. The XMLs (panda.xml, dpg_scene.xml, ...) are committed; only the meshes
# are fetched. Run once after cloning:  bash sim/fetch_assets.sh
set -e
cd "$(dirname "$0")/franka"
if ls assets/*.obj >/dev/null 2>&1; then
  echo "assets already present ($(ls assets | wc -l | tr -d ' ') files)"; exit 0
fi
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie "$tmp"
( cd "$tmp" && git sparse-checkout set franka_emika_panda )
cp -r "$tmp/franka_emika_panda/assets" .
rm -rf "$tmp"
echo "restored $(ls assets | wc -l | tr -d ' ') mesh assets"

#!/usr/bin/env bash
# Clone a pinned CUTLASS into $1
set -euo pipefail

dest="${1:?usage: fetch_cutlass.sh DESTDIR}"
version="${CUTLASS_VERSION:-3.5.1}"
commit="${CUTLASS_COMMIT:-f7b19de32c5d1f3cedfc735c2849f12b537522ee}"

if [ ! -d "$dest/include" ]; then
  git clone --branch "v${version}" --depth 1 \
    https://github.com/NVIDIA/cutlass.git "$dest" >&2
fi
head="$(git -C "$dest" rev-parse HEAD)"
if [ -n "$commit" ] && [ "$head" != "$commit" ]; then
  echo "!! cutlass HEAD $head is not the pinned $commit" >&2
  exit 1
fi
echo "$dest"

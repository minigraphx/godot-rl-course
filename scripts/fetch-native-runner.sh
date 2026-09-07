#!/usr/bin/env bash
#
# Fetch the pinned godot-native-rl addon into the course game project.
#
# The addon ships prebuilt native libraries for macOS, Windows, Linux, iOS,
# Android and Web. They total far more than the rest of this repository, so they
# are not committed — this script downloads the pinned release and verifies it
# against the SHA256 published with that release.
#
# Usage:
#   scripts/fetch-native-runner.sh          # install the pinned version
#   scripts/fetch-native-runner.sh --check  # report status, change nothing
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAME_DIR="$REPO_ROOT/examples/neural_foundations/game"
PIN_FILE="$GAME_DIR/GODOT_NATIVE_RL_VERSION"
ADDON_DIR="$GAME_DIR/addons/godot_native_rl"

die() { printf '\nerror: %s\n' "$1" >&2; exit 1; }

[ -f "$PIN_FILE" ] || die "pin file not found: $PIN_FILE"

# Read key=value pairs from the pin file, ignoring comments and blank lines.
pin() {
  local key="$1" value
  value="$(grep -E "^${key}=" "$PIN_FILE" | head -1 | cut -d= -f2- || true)"
  [ -n "$value" ] || die "$key missing from $(basename "$PIN_FILE")"
  printf '%s' "$value"
}

VERSION="$(pin release_tag)"
SOURCE_REPO="$(pin source_repo)"
ASSET="$(pin addon_asset)"
EXPECTED_SHA="$(pin addon_sha256)"

installed_version() {
  [ -f "$ADDON_DIR/plugin.cfg" ] || { printf 'none'; return; }
  grep -E '^version=' "$ADDON_DIR/plugin.cfg" | head -1 | cut -d'"' -f2
}

CURRENT="$(installed_version)"

if [ "${1:-}" = "--check" ]; then
  printf 'pinned:    %s\n' "$VERSION"
  printf 'installed: %s\n' "$CURRENT"
  [ "$CURRENT" = "${VERSION#v}" ] && { printf 'status:    up to date\n'; exit 0; }
  printf 'status:    out of date — run scripts/fetch-native-runner.sh\n'
  exit 1
fi

if [ "$CURRENT" = "${VERSION#v}" ]; then
  printf 'godot-native-rl %s already installed. Nothing to do.\n' "$VERSION"
  exit 0
fi

command -v curl >/dev/null || die "curl is required"
command -v unzip >/dev/null || die "unzip is required"

# shasum on macOS, sha256sum on most Linux distributions.
if command -v shasum >/dev/null; then SHA_CMD=(shasum -a 256)
elif command -v sha256sum >/dev/null; then SHA_CMD=(sha256sum)
else die "neither shasum nor sha256sum found"; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

URL="https://github.com/${SOURCE_REPO}/releases/download/${VERSION}/${ASSET}"
printf 'Downloading %s …\n' "$ASSET"
curl -fsSL --retry 3 -o "$TMP/$ASSET" "$URL" \
  || die "download failed: $URL"

ACTUAL_SHA="$("${SHA_CMD[@]}" "$TMP/$ASSET" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  die "checksum mismatch for $ASSET
  expected: $EXPECTED_SHA
  actual:   $ACTUAL_SHA
The download was corrupted or the release asset changed. Nothing was installed."
fi
printf 'Checksum verified.\n'

unzip -q "$TMP/$ASSET" -d "$TMP/unpacked" || die "unzip failed"
[ -d "$TMP/unpacked/addons/godot_native_rl" ] \
  || die "unexpected archive layout: addons/godot_native_rl not found"

# Replace wholesale — a partial overlay would leave files from the old version
# behind, and stale .gd files next to a newer binary fail in confusing ways.
rm -rf "$ADDON_DIR"
mkdir -p "$(dirname "$ADDON_DIR")"
mv "$TMP/unpacked/addons/godot_native_rl" "$ADDON_DIR"

printf 'Installed godot-native-rl %s into %s\n' \
  "$VERSION" "${ADDON_DIR#"$REPO_ROOT"/}"
printf 'Platforms: %s\n' \
  "$(ls "$ADDON_DIR/bin" 2>/dev/null | sed -E 's/^libncnn_runner\.([a-z]+)\..*/\1/' | sort -u | tr '\n' ' ')"

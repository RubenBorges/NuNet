#!/usr/bin/env bash

set -euo pipefail

require_command() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Error: required command '$1' was not found in PATH." >&2
		exit 1
	fi
}

cleanup_artifacts() {
	rm -rf "$BUILD_DIR" "$INSTALL_DIR" "$OUT_DIR"
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

readonly BUILD_DIR="build"
readonly INSTALL_DIR="dist"
readonly OUT_DIR="out"
readonly PRESETS_FILE="$SCRIPT_DIR/CMakePresets.json"
readonly CONFIGURE_PRESET="linux-dev"
readonly BUILD_PRESET="dev-build"
readonly INSTALL_PRESET="dev-install"

require_command cmake

if [[ ! -f "$PRESETS_FILE" ]]; then
	echo "Error: CMakePresets.json was not found in $SCRIPT_DIR." >&2
	exit 1
fi

echo "--- Cleaning old build ---"
cleanup_artifacts

echo "--- Configuring with preset '$CONFIGURE_PRESET' ---"
cmake --preset "$CONFIGURE_PRESET"

echo "--- Building with preset '$BUILD_PRESET' ---"
cmake --build --preset "$BUILD_PRESET"

echo "--- Installing with preset '$INSTALL_PRESET' ---"
cmake --build --preset "$INSTALL_PRESET"

echo "--- Done! Executable is in $INSTALL_DIR/bin/ ---"
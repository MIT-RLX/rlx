#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Build the reference driver extension and its helper.
#
# The helper builds anywhere with the macOS SDK. The extension needs the
# DriverKit SDK (full Xcode, not just Command Line Tools) and, to actually load,
# a signing identity holding com.apple.developer.driverkit.transport.pci — which
# Apple grants per team. See docs/egpu.md.
#
#   ./build.sh helper     # just the helper (no entitlement needed)
#   ./build.sh dext       # the extension (needs Xcode + DriverKit SDK)
#   ./build.sh            # both, skipping the dext if the SDK is absent

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

out=${RLX_DEXT_OUT:-build}
identity=${RLX_CODESIGN_IDENTITY:-}
bundle_id=${RLX_DEXT_BUNDLE_ID:-org.rlx.egpu.RlxPciDriver}
target=${1:-all}

mkdir -p "$out"

build_helper() {
  echo "==> helper"
  cc -O2 -Wall -Wextra -o "$out/rlxpci-helper" helper/helper.c \
     -framework CoreFoundation -framework IOKit
  echo "    $out/rlxpci-helper"
}

build_dext() {
  echo "==> driver extension"
  local sdk
  if ! sdk=$(xcrun --sdk driverkit --show-sdk-path 2>/dev/null); then
    echo "    SKIP: no DriverKit SDK (needs full Xcode, not Command Line Tools)" >&2
    return 1
  fi
  echo "    SDK $sdk"

  # iig generates the .h from each .iig; the .cpp files include those headers,
  # which is why they do not parse standalone in an editor. `-D__IIG=1` and the
  # SDK include paths are required — without them iig fails to parse the SDK's
  # own .iig headers, which reads as an error in DriverKit rather than here.
  local iig_flags=(-isysroot "$sdk" -x c++ -std=gnu++17 -D__IIG=1
                   -I"$sdk/System/DriverKit/usr/include"
                   -I"$sdk/System/DriverKit/System/Library/Frameworks")
  for iig in RlxPciDriver.iig RlxPciDriverUserClient.iig; do
    xcrun iig --def "$iig" --header "$out/${iig%.iig}.h" --impl "$out/${iig%.iig}.iig.cpp" \
              -- "${iig_flags[@]}"
  done

  xcrun --sdk driverkit clang++ -std=gnu++17 -O2 -fno-exceptions -fno-rtti \
    -isysroot "$sdk" -I"$out" -I. \
    -target arm64-apple-driverkit \
    -framework DriverKit -framework PCIDriverKit \
    -o "$out/RlxPciDriver" \
    RlxPciDriver.cpp RlxPciDriverUserClient.cpp \
    "$out/RlxPciDriver.iig.cpp" "$out/RlxPciDriverUserClient.iig.cpp"

  # Assemble the .dext bundle.
  local bundle="$out/RlxPciDriver.dext"
  rm -rf "$bundle"
  mkdir -p "$bundle/Contents/MacOS"
  sed "s|\$(PRODUCT_BUNDLE_IDENTIFIER)|$bundle_id|g" Info.plist > "$bundle/Contents/Info.plist"
  mv "$out/RlxPciDriver" "$bundle/Contents/MacOS/RlxPciDriver"

  if [ -n "$identity" ]; then
    codesign --force --sign "$identity" --entitlements RlxPciDriver.entitlements \
             --options runtime "$bundle"
    echo "    signed as $identity"
  else
    echo "    NOT SIGNED: set RLX_CODESIGN_IDENTITY to sign (it will not load unsigned)" >&2
  fi
  echo "    $bundle"
}

case "$target" in
  helper) build_helper ;;
  dext)   build_dext ;;
  all)    build_helper; build_dext || true ;;
  *) echo "usage: $0 [helper|dext|all]" >&2; exit 2 ;;
esac

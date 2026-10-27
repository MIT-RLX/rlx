#!/usr/bin/env bash
# Build the RLX node app (ios/Demo) and install it on attached iPhones and
# iPads, in one command:
#
#   ios/deploy-node.sh                       # every attached iOS device
#   ios/deploy-node.sh "Yellow iPad"         # just these (devicectl names)
#   BLE=1 ios/deploy-node.sh                 # + mesh links over Bluetooth LE
#   ios/deploy-node.sh --no-framework        # app only: the Rust side is unchanged
#
# The node is built with Metal / ANE and `mode = "model"` (what `cargo rlx
# dist` runs), and only for devices: no simulator slices, so one Rust build
# rather than three. The app is built once for any iOS device and installed on
# each, rather than built per device. Rust rebuilds are incremental across
# runs (target/aarch64-apple-ios/release), and so is the app's build
# (target/ios-app; ios/build is rebuilt from scratch each time). The Xcode
# project is generated from ios/Demo/project.yml with xcodegen when it is
# missing or older.
#
# Signing: DEVELOPMENT_TEAM, else the team the last build used (kept in
# ~/.config/rlx/ios-team), else the team
# of this Mac's Apple Development certificates when there is just one. Extra arguments for build-xcframework.sh go after `--`.
set -euo pipefail

cd "$(dirname "$0")"
IOS="$PWD"
FRAMEWORK=1
DEVICES=()
XCF_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-framework) FRAMEWORK=0; shift ;;
    --) shift; XCF_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) DEVICES+=("$1"); shift ;;
  esac
done

secs() { echo $(( $(date +%s) - $1 ))s; }

if [[ ${#DEVICES[@]} -eq 0 ]]; then
  json=$(mktemp)
  xcrun devicectl list devices --json-output "$json" >/dev/null
  # Attached = known to CoreDevice now; `unavailable` is a paired device
  # that's away.
  while IFS= read -r d; do DEVICES+=("$d"); done < <(python3 -c '
import json, sys
for d in json.load(open(sys.argv[1]))["result"]["devices"]:
    if d["hardwareProperties"].get("platform") == "iOS" \
            and d["connectionProperties"].get("tunnelState") != "unavailable":
        print(d["deviceProperties"]["name"])' "$json")
  rm -f "$json"
  [[ ${#DEVICES[@]} -gt 0 ]] || { echo "no attached iOS device (xcrun devicectl list devices)" >&2; exit 1; }
fi

DD="$IOS/../target/ios-app"
mkdir -p "$DD"
# The teams of this Mac's valid (not revoked) Apple Development certificates.
dev_teams() {
  local valid
  valid=$(security find-identity -v -p codesigning | awk '/Apple Development/ {print $2}')
  security find-certificate -a -c "Apple Development" -p -Z 2>/dev/null | python3 -c '
import re, subprocess, sys
valid = set(sys.argv[1].split())
teams = set()
for sha, pem in re.findall(r"SHA-1 hash: (\w+).*?(-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----)",
                           sys.stdin.read(), re.S):
    if sha in valid:
        subj = subprocess.run(["openssl", "x509", "-noout", "-subject"], input=pem,
                              capture_output=True, text=True).stdout
        m = re.search(r"OU ?= ?([A-Z0-9]{10})", subj)
        if m:
            teams.add(m.group(1))
print("\n".join(sorted(teams)))' "$valid"
}
# Kept outside target/, which a `cargo clean` empties.
TEAM_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/rlx/ios-team"
TEAM=${DEVELOPMENT_TEAM:-$(cat "$TEAM_FILE" 2>/dev/null || true)}
if [[ -z $TEAM ]]; then
  teams=$(dev_teams)
  if [[ $(echo "$teams" | grep -c .) == 1 ]]; then
    TEAM=$teams
  else
    echo "set DEVELOPMENT_TEAM (remembered after the first build); this Mac signs for:" >&2
    echo "${teams:-  (no Apple Development certificate)}" | sed 's/^/  /' >&2
    exit 1
  fi
fi

echo "==> devices: ${DEVICES[*]}  (team $TEAM)"
if [[ $FRAMEWORK == 1 ]]; then
  t=$(date +%s)
  "$IOS/build-xcframework.sh" --apple --model --platforms ios --device-only \
    "${XCF_ARGS[@]+"${XCF_ARGS[@]}"}"
  echo "==> node framework built in $(secs "$t")"
fi

PROJ="$IOS/Demo/RlxDemo.xcodeproj"
if [[ ! -d $PROJ || $IOS/Demo/project.yml -nt $PROJ ]]; then
  command -v xcodegen >/dev/null || { echo "xcodegen needed for $PROJ: brew install xcodegen" >&2; exit 1; }
  (cd "$IOS/Demo" && xcodegen generate --quiet)
fi

t=$(date +%s)
log="$DD/xcodebuild.log"
if ! xcodebuild -project "$PROJ" -scheme RlxDemo -configuration Release \
    -destination generic/platform=iOS -derivedDataPath "$DD" -allowProvisioningUpdates \
    DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build > "$log" 2>&1; then
  grep -E "error:|BUILD FAILED" "$log" | head -20 >&2
  echo "app build failed — full log: $log" >&2
  exit 1
fi
mkdir -p "$(dirname "$TEAM_FILE")" && echo "$TEAM" > "$TEAM_FILE"
echo "==> app built in $(secs "$t")"

APP="$DD/Build/Products/Release-iphoneos/RlxDemo.app"
failed=0
for d in "${DEVICES[@]}"; do
  if xcrun devicectl device install app --device "$d" "$APP" >/dev/null 2>&1; then
    echo "==> installed on $d"
  else
    echo "install on $d failed (xcrun devicectl device install app --device \"$d\" $APP)" >&2
    failed=1
  fi
done
exit $failed

#!/bin/bash
# Three real game instances in one co-op session, against a private server
# on this machine, with a leave and a rejoin in the middle:
#
#   host    rings the Beckoning Bell twice, waits for a cooperator to
#           leave, rings again (a ring keeps searching, so one can summon
#           both cooperators, in either order: the verdict allows for it)
#   guest1  rings the Small Resonant Bell, plays 90 s in the host's world,
#           goes home with the Silencing Blank ("Return to Your Own World"),
#           rings again from home and is summoned back
#   guest2  rings and stays in the host's world through all of it
#
# The verdict reads the three logs and the server's: guest2 never loses the
# room while guest1 leaves and comes back, the host hears every join and the
# leave, and nobody's game summons into a room the server no longer has.
#
#   NET_SEED=<dir with userdata0001 + userdata0010> tools/nettest/game_trio.sh
#
# The seed is a save at a lamp outdoors with the bells in the inventory (the
# rows below: Beckoning 155, Small Resonant 157, Silencing Blank 161 in a
# 187-row list); it is copied into each
# instance's data folder, the same character three times under three online
# ids. The guests are test clients (BBHOST_RENDER_MIN, no buffer shadow) so
# three instances fit one GPU; the host renders. Displays :78-:80 (Xvfb).
# The server is a copy of NET_SERVER_TREE (the private server package)
# started by tools/nettest/bench.py --serve on the game's ports 18671/20443,
# STUN and relay on 127.0.0.2 (3478, 4000-4199): stop any other local
# server first. Each instance gets its own config folder (BBHOST_CONFIG_DIR)
# and a config naming only this machine.
set -u
cd "$(dirname "$0")/../.."
R=$PWD
OUT=${NET_OUT:-$R/build/nettest-game}
TREE=${NET_SERVER_TREE:?NET_SERVER_TREE: the private server package directory}
SEED=${NET_SEED:?NET_SEED: a folder with userdata0001 and userdata0010}
APP0=${BBHOST_APP0:?BBHOST_APP0: the game folder that contains dvdroot_ps4}
rm -rf "$OUT"; mkdir -p "$OUT"

# ---- the server -------------------------------------------------------------
python3 tools/nettest/bench.py --server-tree "$TREE" --serve --http-port 18671 --ssinfo-port 20443 \
  --stun-port 3478 --relay-ports 4000-4199 --work "$OUT/server" > "$OUT/server-run.log" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do grep -q "server up" "$OUT/server-run.log" && break; sleep 1; done
grep -q "server up" "$OUT/server-run.log" || { echo "server did not start"; cat "$OUT/server-run.log"; kill $SRV; exit 1; }

# ---- the instances ----------------------------------------------------------
role_cfg() {  # role online_id p2p_port data
  local d="$OUT/$1"; mkdir -p "$d/cfg"
  cat > "$d/bbhost.toml" <<EOF
[paths]
app0 = "$APP0"
data = "$4"
tmp = "$R/tmp"
eboot = "$R/eboot-109-decrypted.bin"

[online]
host = "127.0.0.1"
scheme = "http"
verify_tls = false
online_id = "$2"
np_server = "http://127.0.0.1:18671"
# A fresh config folder (BBHOST_CONFIG_DIR) gets require_account = true and
# the live server as auth_server; this layer keeps both on this machine (a
# signed-out game never goes online).
require_account = false
auth_server = "http://127.0.0.1:18671"
stun_server = "127.0.0.2:3478"
p2p_port = $3

[video]
width = 1920
height = 1080
fps_cap = 30

[startup]
skip_intro = true
EOF
  cat > "$d/run.sh" <<EOF
#!/bin/bash
export SDL_AUDIO_DRIVER=\${SDL_AUDIO_DRIVER:-dummy} BBHOST_CONFIG_DIR="$d/cfg"
exec "$R/build/bbhost" --config "$d/bbhost.toml" "\$@"
EOF
  chmod +x "$d/run.sh"
  mkdir -p "$4/saves/SPRJ0005"
  cp -p "$SEED/userdata0001" "$SEED/userdata0010" "$4/saves/SPRJ0005/"
}
role_cfg host Hunter 9307 "${NET_DATA_HOST:-$R/tmp/np-host-data}"
role_cfg guest1 Hunter2 9308 "${NET_DATA_GUEST1:-$R/tmp/np-guest-data}"
role_cfg guest2 Hunter3 9309 "${NET_DATA_GUEST2:-$R/tmp/np-third-data}"

export BBHOST_GAME_FPS=60 BBHOST_NP_TRACE=1 BBHOST_NP_TEST=probe BBHOST_TEST_INSIGHT=5 BBHOST_RES=960x540
# Open the inventory, scroll to row $1, use the item (the Use popup's YES).
use() { echo "key:Escape waitnew:count.24.cells.6 sleep:1 click:0 waitnew:count.187.cells.1 sleep:1 move:1150,750 sleep:1 wheelto:$1 sleep:1 shot:$2 keyhold:Return,0.6 waitnew:count.2.cells.1.lines.5 sleep:1 click:0 sleep:4 shot:$2-used"; }
WORLD="title:online sleep:4 clicknow:0 waitlog:area.glare.luminance sleep:25"
G1="$WORLD $(use 157 g1-bell) waitnew:np:.joined.room sleep:90 shot:g1-coop $(use 161 g1-blank) sleep:1 click:1 sleep:3 shot:g1-blank-confirm click:0 sleep:4 shot:g1-blank-ret waitn:3,np.test:.the.world.is.up sleep:30 shot:g1-home $(use 157 g1-bell2) waitn:2,np:.joined.room sleep:90 shot:g1-coop2 sleep:60 shot:g1-end"
G2="$WORLD $(use 157 g2-bell) waitnew:np:.joined.room sleep:60 shot:g2-coop sleep:700 shot:g2-end"
HOST="$WORLD $(use 155 h-bell) waitnew:np:.member.*joined sleep:45 shot:h-coop1 $(use 155 h-bell2) waitnew:np:.member.*joined sleep:60 shot:h-coop2 waitnew:np:.member.*left sleep:30 shot:h-after-leave $(use 155 h-bell3) waitnew:np:.member.*joined sleep:120 shot:h-coop3 sleep:60 shot:h-end"
drive() {  # role display steps... (test clients for the guests)
  local role=$1 disp=$2; shift 2
  local extra=()
  [ "$role" != host ] && extra=(BBHOST_RENDER_MIN=1 BBHOST_BUFFER_SHADOW=0)
  env "${extra[@]}" MENU_DRIVE_OUT="$OUT/$role/drive" DISPLAY_DRIVE=$disp BBHOST_BIN="$OUT/$role/run.sh" \
    MENU_DRIVE_ONLINE_CONFIG="$OUT/$role/bbhost.toml" MENU_DRIVE_TIMEOUT=1700 \
    tools/menu_drive.sh "$@" > "$OUT/$role/drive.log" 2>&1
}
drive guest1 :79 $G1 &
P1=$!
sleep 40
drive guest2 :80 $G2 &
P2=$!
sleep 40
drive host :78 $HOST &
P3=$!
wait $P1 $P2 $P3
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

python3 tools/nettest/game_verdict.py "$OUT"

# Online play

Bloodborne's online features - co-op, invasions, summon signs, messages,
bloodstains, the play log behind ghosts - are built on the PS4's
PlayStation Network libraries (NP Matching2, Signaling, WebAPI, Auth) and
FromSoftware's own servers. bbhost implements those libraries against a
private server, so the game's online code runs unchanged and bbhost players
can play with each other.

PS4 cross-play is not a goal: bbhost players meet only other bbhost players.

## Connecting

The setup window chooses the server (the public one, a playtest server, or
your own) and links an account. In `bbhost.toml`, the public server's
settings, which are the defaults:

```toml
[online]
host = "thehuntersdream.com"   # replaces FromSoftware's hosts in the game's requests
scheme = "https"               # the public server takes only https
verify_tls = true              # and its certificate is checked
require_account = true         # stay offline until this PC is signed in
auth_server = "https://thehuntersdream.com"   # where the account calls go
```

A configuration written by an older bbhost that still sends the public server
plain http is updated to these settings at the next start.

Accounts have no passwords: a PC is linked to an account (with Discord, or
with an account held by that PC plus a recovery code), and the account's name
is the in-game online name. F10 has the same account screen in game.

Each time the setup window opens with an account, it asks the server whether
the sign-in still holds, without making the window wait, and says what it
found on the Game page's Online card and under Play: signed in,
expired (or this PC was signed out elsewhere: log in with Discord again, or
recover an account kept on this PC), or not checked when the server did not
answer (with a Retry). A sign-in the server refuses keeps the game offline
instead of asking again all session; one refused while the game runs is said
once over the game, and F10's Game page has the way back in.

## How a session connects

- **Matchmaking**: the game's Matching2 calls (create, search and join
  rooms, member events) go to the server's session service. The game's own
  rules decide who can be summoned or invade; bbhost only carries them.
- **Peer to peer**: players exchange game traffic over UDP directly, as on
  the console. Each instance has its own P2P port (`online.p2p_port`, default
  9307; a second instance on one machine takes the next free one).
- **NAT**: the P2P port learns its public address from a STUN server
  (`online.stun_server`) and both sides punch holes towards each other. When
  that fails, the server can relay the traffic.

## Beyond the console

- **Five players**: the late areas (Mensis, Mergo's Loft, the Nightmare
  Frontier, The Old Hunters' areas) take two invaders, as the chalice
  dungeons do. A PC enhancement, on by default; switch it off for the game
  as it shipped.
- **Faster play-log uploads**: the game sends its play log every 30 seconds
  instead of every five minutes, so the server's map and profiles stay
  current (`online.playlog_upload_seconds`; the server asks for no less than
  30).
- **Shorter summon timeouts**: a summon sign that cannot be answered gives up
  after 30 seconds instead of three minutes (`online.sign_timeout_seconds`).
- **Full Phantom HP**: a cooperator or invader keeps their full max HP in the
  host's world instead of the game's 70%. It is each visitor's own setting:
  the host's changes nothing for the hunters in its world, and every game in
  the session shows a visitor's HP as the visitor's own game has it. A PC
  enhancement, on by default; switch it off for the game as it shipped.
- **Modded worlds stay apart**: sessions with gameplay plugins (the
  randomizer, the boss rush, mutators) announce their rules to the server,
  which keeps their messages, bloodstains, signs and statistics separate from
  vanilla players. A player can still be summoned into a randomizer host's
  world and play by the host's rules (see [plugins.md](plugins.md)).

## Running a server

The server is a separate project. Point `online.host` (and, if the account
service lives elsewhere, `online.auth_server`) at your own instance; with
`scheme = "http"` a server on the local network needs no certificate.

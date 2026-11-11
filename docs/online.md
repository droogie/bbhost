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
  that fails, the server can relay the traffic. The setup window's connection
  test (below) tells which applies to this PC.

## The connection test

The setup window's Game page has a Connection card. **Test connection**
checks whether other players can reach this PC directly, or whether its
co-op and PvP go through the server's relay. It runs from the game's own P2P
port, so it needs the game closed (Play waits for it), and it takes a few
seconds.

- It sends STUN requests from the P2P port to the server's two test ports
  (3479 and 3480 at the STUN server's host; `online.nat_test_ports` for a
  server of your own) and reads the public address and port each one saw:
  - **Direct connections: yes, your router keeps one port** - the same public
    port at both: other players can connect to this PC directly.
  - **You'll use the relay: your router changes ports per destination** - a
    different port (or address) at each, as symmetric and many carrier-grade
    NATs do: a direct connection fails, and the relay carries the session.
  - **UDP is blocked on this network** - no answer at all: online play will
    not work from this network (a firewall, a VPN, a school or work network).
- **Incoming**, the port check: one more request asks the server to answer
  from a third port, which the game never sent anything to. When that answer
  arrives, the router and this PC's firewall let in a packet nobody asked for
  (a forwarded port, or a router that does not filter by port), so even
  players whose routers change ports can connect directly. Closed is common
  and fine: those players reach you through the relay instead. A server
  without the check says so.
- Signed in, the server classifies what it saw from your address and keeps
  the result for your account, as the website's account page test does, and
  a server that relays only the players who need it relays you when the
  result says so. Not signed in, the card shows this PC's own reading, not
  saved. The server takes 10 tests an hour per account.
- The card says what the result means, the port the test ran from, and when
  your account's last test ran (also one taken on the website). F10's Game
  page repeats the verdict when the setup window opened before the game. The
  log has one line per test (`connection test: ...`), with no addresses.

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

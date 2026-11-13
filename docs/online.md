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
seconds - up to about six when nothing comes back, sending again meanwhile
so that a security program holding the first packets while it asks about
bbhost.exe does not spoil the result.

The card says the result in one headline with its status (a check, a
warning or a problem), one sentence on what it means for co-op and PvP, and
the one thing to do when there is something:

| Headline | What it means | What to do |
|---|---|---|
| **Your network allows direct connections** | the same public port at both test ports, with the relay on (the default): co-op and PvP go through the relay all the same | to connect directly, turn the relay off (the switch under the result) - with incoming closed, after the steps to forward the port; signed out, sign in to play online |
| **Online play works directly** | the same, with the relay off (or on a server without the switch): other players connect to this PC without the relay | nothing; with incoming closed, the steps to forward the port (below) |
| **Online play works through the relay** | a different port (or address) at each, as symmetric and many carrier-grade NATs do: sessions go through the relay, with a little more delay | nothing (not signed in: sign in and test again, so the server knows to relay you) |
| **Something is blocking bbhost** | nothing came back: something on this PC (a firewall or antivirus) or the network stops bbhost's packets or the answers | sign in and test again to find out which - the server tells |
| **Online play is blocked** | signed in, the server saw none of bbhost's packets | allow bbhost.exe in the firewall or antivirus; if the website's test fails too, the network blocks online play (school and work networks can) |
| **Answers aren't reaching bbhost** | the server heard bbhost, but none of its answers came back - most likely a firewall or antivirus on this PC | allow bbhost.exe in the firewall or antivirus |
| **Something on this PC is blocking bbhost** | the system refused every packet bbhost tried to send (a permission error) | allow bbhost.exe in the firewall or antivirus |
| **This PC couldn't send anything** | the system refused every packet for another reason (no network) | check the internet connection, then the firewall |
| **A VPN or proxy may be in the way** | the test ports answered, but the server saw none of it from the address the sign-in came from (a VPN or a proxy for web traffic) | turn off the VPN or proxy |
| **The test was incomplete** | one test port answered: a lost packet, most likely | test again |
| **The game's port is in use** / **The test couldn't use UDP port ...** / **The test server wasn't found** | the test could not run | close the game, choose another P2P port, check the connection |

- **Incoming**, the port check: one more request asks the server to answer
  from a third port, which the game never sent anything to. When that answer
  arrives, the router and this PC's firewall let in a packet nobody asked for
  (a forwarded port, or a router that does not filter by port), so even
  players whose routers change ports can connect directly. Closed, those
  players reach you through the relay instead, and the card gives the steps
  that let them in: forward the UDP port to this PC in the router's settings
  - with the PC's address on its network, the one the test's requests left
  from - allow bbhost.exe in Windows Firewall (Private and Public), and test
  again. Behind a router that changes ports, forwarding does not help and
  the card leaves it out. A server without the check says so.
- **Details** opens the technical lines: the port the test ran from and the
  public port the server saw, the port check, when, whether the result was
  kept for your account (or is this PC's own reading), the server's verdict
  beside this PC's when they differ, refused sends, and the PC's address on
  its network.
- Signed in, the server classifies what it saw from your address and keeps
  the result for your account, as the website's account page test does, and
  a server that relays only the players who need it relays you when the
  result says so. Not signed in, the card shows this PC's own reading, not
  saved. The server takes 10 tests an hour per account. Before a test runs
  in this window, the card shows your account's last one (also one taken on
  the website).
- F10's Game page repeats the card's headline, in its colour, when the setup
  window opened before the game. The log has one line per test
  (`connection test: ...`): the result, how many requests went out and how
  many the system refused (with its first error), and no addresses.

### The P2P port

Under the result, **P2P port** (`online.p2p_port`, 9307 by default) is the
UDP port other players reach this PC on: the one to forward on the router.
It takes 1024 to 65535, not the server's test ports, and not a port another
program has (two copies of bbhost on one PC need different ports). The next
test runs from it at once; Save or Play keeps it, and the game opens it as it
starts, so a change made while the game runs applies from its next start.

### The relay switch

Signed in, **Relay** is your account's own setting, the same as the
website's account page has: **On** (the default) works on every network, as
co-op and PvP go through the server's relay; **Off** connects directly, which
needs your router to let other players in. Players who keep the relay still
reach you through it, and everyone who plays from your connection (your home
network) needs it off for it to apply. Turning it off after a test that
showed a direct result with incoming open takes one click; otherwise the
card says why it may not work and gives the port-forwarding steps, and asks
again (**Turn it off anyway**, or **Keep the relay**). Turning it back on is
one click. The change goes to the server at once, and the card reads it back
from the server. With the relay off while other players cannot get in, the
card's headline says so - **Other hunters may not reach you with the relay
off** - with the steps and the switch.

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

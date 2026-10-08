#!/usr/bin/env python3
"""Announce a GitHub release in Discord, as the server's bot.

The release's own notes are the message: the summary and the points written
for players come first in them, so they lead the post. Pull-request lists
(--generate-notes) are trimmed to their titles, the "Full Changelog" line is
dropped, and long notes are cut at a line with a link to the rest. The
downloads go in a field of their own. Drafts and pre-releases are skipped.

    tools/announce_release.py [TAG]       the release with this tag, else the latest

Environment:
    GITHUB_REPOSITORY      owner/name (set in GitHub Actions)
    GH_TOKEN               a token to read the release (optional for a public repo)
    DISCORD_BOT_TOKEN      the bot's token
    DISCORD_CHANNEL_ID     the channel to post in
    DRY_RUN=1              print the message instead of posting it
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
DESC_MAX = 3500          # Discord allows 4096 in an embed description
FIELD_MAX = 1024


def github(path: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "bbhost-announce"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(API + path, headers=headers), timeout=20) as r:
        return json.load(r)


def notes(body: str, url: str) -> str:
    lines = []
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        if re.match(r"\s*\*\*Full Changelog\*\*", line):
            continue
        # A generated pull-request line: keep its title.
        line = re.sub(r"\s+by @[\w-]+ in https://github\.com/\S+/pull/\d+\s*$", "", line)
        # No mention can ping anyone (allowed_mentions too), and no stray escapes.
        line = line.replace("@everyone", "@​everyone").replace("@here", "@​here")
        lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) <= DESC_MAX:
        return text
    more = f"\n\n…[read the full notes]({url})"
    cut, out = DESC_MAX - len(more), []
    for line in text.split("\n"):
        if sum(len(x) + 1 for x in out) + len(line) > cut:
            break
        out.append(line)
    return "\n".join(out).rstrip() + more


def downloads(assets: list) -> str:
    kinds = [("Windows", r"^bbhost-win-.*\.zip$"), ("Linux", r"^bbhost-linux-.*\.tar\.gz$"),
             ("Steam Deck", r"^bbhost-steamdeck-.*\.tar\.gz$")]
    out = []
    for label, pattern in kinds:
        for a in assets:
            if re.match(pattern, a.get("name", "")):
                out.append(f"[{label}]({a['browser_download_url']})")
                break
    return " · ".join(out)[:FIELD_MAX]


def message(rel: dict) -> dict:
    tag, url = rel["tag_name"], rel["html_url"]
    embed = {"title": f"bbhost {tag} is out", "url": url,
             "description": notes(rel.get("body", ""), url) or f"[Release notes]({url})",
             "color": 0x8B1E1E}
    fields = []
    if (d := downloads(rel.get("assets", []))):
        fields.append({"name": "Downloads", "value": d, "inline": False})
    if fields:
        embed["fields"] = fields
    if rel.get("published_at"):
        embed["timestamp"] = rel["published_at"]
    return {"embeds": [embed], "allowed_mentions": {"parse": []}}


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "droogie/bbhost")
    tag = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    rel = github(f"/repos/{repo}/releases/tags/{tag}" if tag else f"/repos/{repo}/releases/latest")
    if rel.get("draft") or rel.get("prerelease"):
        print(f"{rel['tag_name']}: a draft or pre-release, not announced")
        return 0
    payload = message(rel)
    if os.environ.get("DRY_RUN") == "1":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    token, channel = os.environ.get("DISCORD_BOT_TOKEN", ""), os.environ.get("DISCORD_CHANNEL_ID", "")
    if not token or not channel:
        print("DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID are needed to post", file=sys.stderr)
        return 1
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel}/messages", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                 "User-Agent": "DiscordBot (https://github.com/droogie/bbhost, 1) announce-release"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"{rel['tag_name']}: announced (HTTP {r.status})")
    except urllib.error.HTTPError as e:
        print(f"{rel['tag_name']}: Discord refused the post: HTTP {e.code} {e.read()[:300]!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

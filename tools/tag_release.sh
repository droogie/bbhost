#!/bin/bash
# Tag a release with its notes, which the release workflow publishes and the
# Discord bot announces (.github/workflows/release.yml).
#   tools/tag_release.sh v0.2.17 notes.md [COMMIT]     then: git push <remote> v0.2.17
# notes.md is the release notes in Markdown, for players. The tag's message is
# "bbhost v0.2.17", a blank line, then the notes exactly as written: git's
# default cleanup would drop every line starting with '#' (Markdown headings).
set -eu
tag=${1:?usage: tools/tag_release.sh vX.Y.Z NOTES.md [COMMIT]}
notes=${2:?usage: tools/tag_release.sh vX.Y.Z NOTES.md [COMMIT]}
commit=${3:-HEAD}
fail() { echo "tag_release: $*" >&2; exit 1; }
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "$tag is not vX.Y.Z"
[ -f "$notes" ] || fail "no notes file $notes"
grep -q '[^[:space:]]' "$notes" || fail "$notes is empty"
if head -n 1 "$notes" | grep -q '^bbhost v'; then fail "$notes starts with a title line: the script adds it, leave it out"; fi
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && fail "$tag exists already"
git rev-parse --verify "$commit^{commit}" >/dev/null || fail "no commit $commit"
msg=$(mktemp); trap 'rm -f "$msg"' EXIT
{ printf 'bbhost %s\n\n' "$tag"; cat "$notes"; } > "$msg"
git tag -a --cleanup=verbatim -F "$msg" "$tag" "$commit"
echo "tagged $tag at $(git rev-parse --short "$commit"); the release notes are:"
git tag -l --format='%(contents:body)' "$tag"
echo "push it to release: git push <remote> $tag"

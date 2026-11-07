#!/usr/bin/env python3
"""Checks host/options.cpp's settings table against its enum.

g_set is indexed by the Setting enum (kWindowMode, kVsync, ...), so its entries
must come in the enum's order: an entry added at the wrong place shifts every
setting after it, and each row then shows and saves its neighbour's (the HD
motion blur rows and the DLSS preset did, 2026-11-04). Each entry's key is its
enum name in snake case (kHdMotionBlurRes - "hd_motion_blur_res"); this
compares the two lists in order and exits 1 at the first difference.

    python tools/check_options_order.py [src/host/options.cpp]
"""
import re
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'src/host/options.cpp'
    s = open(path, encoding='utf-8').read()
    enum = re.search(r'enum[^{;]*\{\s*(kWindowMode\b.*?)\bkSettingCount\b', s, re.S)
    table = re.search(r'Setting g_set\[kSettingCount\] = \{(.*?)\n\};', s, re.S)
    if not enum or not table:
        print('check_options_order: the enum or the table was not found in', path)
        return 1
    names = re.findall(r'\bk([A-Za-z0-9]+)\s*,', enum.group(1))
    keys = re.findall(r'^\s*\{"([a-z0-9_]+)",\s*"', table.group(1), re.M)
    norm = lambda x: x.replace('_', '').lower()
    for i, (n, k) in enumerate(zip(names, keys)):
        if norm(n) != norm(k):
            print('check_options_order: entry %d is "%s" where the enum has k%s' % (i, k, n))
            return 1
    if len(names) != len(keys):
        print('check_options_order: %d enum names, %d table entries' % (len(names), len(keys)))
        return 1
    print('check_options_order: %d settings, the table in the enum\'s order' % len(keys))
    return 0


if __name__ == '__main__':
    sys.exit(main())

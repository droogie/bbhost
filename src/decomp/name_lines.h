#pragma once

// The decomp's install report in lines the log holds (decomp.cpp): names
// joined by ", " into parts of at most `width` characters, in their order,
// each name whole and in exactly one part; a name longer than `width` gets a
// part of its own. One host_log line holds 1023 characters, which the report
// of 200 functions passes, so its end - the functions left the game's - never
// showed when it was one line.

#include <cstddef>
#include <string>
#include <vector>

inline std::vector<std::string> decomp_name_lines(const std::vector<std::string>& names, std::size_t width) {
    std::vector<std::string> lines;
    std::string line;
    for (const std::string& name : names) {
        if (!line.empty() && line.size() + 2 + name.size() > width) {
            lines.push_back(line);
            line.clear();
        }
        if (!line.empty()) line += ", ";
        line += name;
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

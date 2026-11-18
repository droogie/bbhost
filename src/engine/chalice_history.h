// Read the live event instances. Root dungeons select several room scripts;
// their running event arguments identify the actual boss and layer without
// guessing from an NPC model or scanning every possible room on disk.
#pragma once
#include "bbhost/engine/sprj/emk_system.hpp"
#include <map>
#include <set>
#include <vector>

namespace chalice_events {
struct Source { std::uint32_t name_id, defeated, begun; int layer; };
template<class Read>
bool sources(std::uint64_t system, std::uint64_t interpreter_vtable, const Read& read, std::vector<Source>& out) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, Source> found;
    std::set<std::uint64_t> visited;
    for (std::uint64_t lane : {0u, 8u}) {
        std::uint64_t node = 0;
        if (!system || !read(system + lane, &node)) return false;
        while (node) {
            if (visited.size() >= 32768 || !visited.insert(node).second) return false;
            bb::SprjEmkEventCore core{};
            if (!read(node, &core)) return false;
            if (reinterpret_cast<std::uint64_t>(core.vtable) == interpreter_vtable && !core.is_retired() &&
                (core.event_id == 12906806 || core.event_id == 12906795)) {
                struct Arguments { std::uint32_t size, padding; std::uint64_t data; } arguments{};
                if (!read(node + offsetof(bb::SprjEmkEventIns, argument_size), &arguments)) return false;
                if (arguments.size >= 20 && arguments.size <= 256) {
                    std::uint32_t values[5]{};
                    if (!arguments.data || !read(arguments.data, &values)) return false;
                    // entity, health-bar name, defeat flag, fight-begun flag,
                    // battle-finished flag. Watchers adds its other actors.
                    if (values[1] && values[1] <= 10000000 && values[2] >= 12901800 && values[2] <= 12901803 &&
                        values[3] >= 12900000 && values[3] < 12910000) {
                        Source s{values[1], values[2], values[3], static_cast<int>(values[2] - 12901800 + 1)};
                        found[{s.defeated, s.name_id}] = s;
                    }
                }
            }
            node = reinterpret_cast<std::uint64_t>(core.next);
        }
    }
    out.clear();
    for (const auto& [key, value] : found) out.push_back(value);
    return true;
}
}

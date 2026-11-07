#include "engine/phantom_hp.h"

#include "core/config.h"
#include "engine/paramdef.h"
#include "engine/params.h"
#include "log.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// The rows the game keeps on a hunter in another's world.
constexpr std::uint32_t kGuestRows[] = {9006, 9026, 9036, 9046, 9056};
constexpr float kGameRate = 0.7f, kWholeRate = 1.0f;

}  // namespace

void phantom_hp_params_loaded() {
    if (!config().full_phantom_hp) return;
    const ParamDefinition* def = params_definition("SpEffectParam");
    const ParamField* field = def ? def->field("maxHpRate") : nullptr;
    if (!field || field->type != "f32" || field->bits) {
        host_log("Full Phantom HP: SpEffectParam's rows are not laid out as its definition says; a summoned hunter keeps the "
                 "game's 70%% of their HP");
        return;
    }
    std::string written, left;
    for (const std::uint32_t id : kGuestRows) {
        std::size_t bytes = 0;
        auto* row = static_cast<std::uint8_t*>(params_row("SpEffectParam", id, &bytes));
        char item[64];
        if (!row || (bytes && bytes < field->offset + sizeof(float))) {
            std::snprintf(item, sizeof(item), "%s%u (no such row)", left.empty() ? "" : ", ", id);
            left += item;
            continue;
        }
        float rate = 0.0f;
        std::memcpy(&rate, row + field->offset, sizeof(rate));
        if (rate != kGameRate) {
            std::snprintf(item, sizeof(item), "%s%u at %.2f", left.empty() ? "" : ", ", id, static_cast<double>(rate));
            left += item;
            continue;
        }
        std::memcpy(row + field->offset, &kWholeRate, sizeof(kWholeRate));
        written += (written.empty() ? "" : ", ") + std::to_string(id);
    }
    if (!written.empty())
        host_log("Full Phantom HP: SpEffectParam %s maxHpRate 0.70 -> 1.00 (a summoned hunter's max HP stays whole)",
                 written.c_str());
    if (!left.empty()) host_log("Full Phantom HP: SpEffectParam %s left as they are (not the game's 0.70)", left.c_str());
}

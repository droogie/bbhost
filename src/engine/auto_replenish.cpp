#include "engine/auto_replenish.h"

#include "core/config.h"
#include "engine/paramdef.h"
#include "engine/params.h"
#include "log.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

// The consumables that refill as Blood Vials and Quicksilver Bullets do
// (engine/auto_replenish.h names them).
constexpr std::uint32_t kGoods[] = {1100, 1101, 1110, 1120, 1200, 1201, 1210, 1220, 1230, 1240,
                                    1250, 1251, 1260, 1270, 1300, 1320, 1330, 1400, 1410, 2030};

}  // namespace

void auto_replenish_params_loaded() {
    if (!config().auto_replenish) return;
    const ParamDefinition* def = params_definition("EquipParamGoods");
    const ParamField* field = def ? def->field("isAutoReplenish") : nullptr;
    if (!field || field->bits != 1) {
        host_log("Auto-Replenish Items: EquipParamGoods' rows are not laid out as its definition says; the "
                 "consumables refill as the game has them");
        return;
    }
    std::string written, already, missing;
    for (const std::uint32_t id : kGoods) {
        std::size_t bytes = 0;
        auto* row = static_cast<std::uint8_t*>(params_row("EquipParamGoods", id, &bytes));
        if (!row || (bytes && bytes < field->offset + field->bytes)) {
            missing += (missing.empty() ? "" : ", ") + std::to_string(id);
            continue;
        }
        if (paramdef_read(*field, row) != "0") {
            already += (already.empty() ? "" : ", ") + std::to_string(id);
            continue;
        }
        if (paramdef_write(*field, row, "1")) written += (written.empty() ? "" : ", ") + std::to_string(id);
    }
    if (!written.empty())
        host_log("Auto-Replenish Items: EquipParamGoods %s isAutoReplenish 0 -> 1 (they refill from storage as Blood "
                 "Vials do)",
                 written.c_str());
    if (!already.empty())
        host_log("Auto-Replenish Items: EquipParamGoods %s already isAutoReplenish 1", already.c_str());
    if (!missing.empty()) host_log("Auto-Replenish Items: EquipParamGoods has no row %s", missing.c_str());
}

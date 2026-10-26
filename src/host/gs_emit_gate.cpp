// A strip GS's emit gate (host/gs_emit_gate.h).
#include "host/gs_emit_gate.h"

#include <algorithm>
#include <cstring>
#include <vector>

using namespace gcn;

namespace {

bool starts(const char* m, const char* p) { return m && std::strncmp(m, p, std::strlen(p)) == 0; }

// The scalar registers (codes below 128: SGPRs, VCC, M0, EXEC) an instruction
// may write - more than it does where unsure, which can only lose a gate.
void scalar_writes(const Inst& in, std::vector<int>& out) {
    out.clear();
    const char* m = mnemonic(in);
    const auto pair = [&](int r) {
        out.push_back(r);
        out.push_back(r + 1);
    };
    switch (in.enc) {
    case Enc::SOP2:
    case Enc::SOP1:
    case Enc::SOPK:
        if (in.dst < 128) {
            if (!m || std::strstr(m, "64")) pair(in.dst);  // a 64-bit result: the pair
            else out.push_back(in.dst);
        }
        if (!m || std::strstr(m, "saveexec")) pair(kExecLo);
        break;
    case Enc::SMRD: {
        int n = 0;
        if (starts(m, "s_load_dword") || starts(m, "s_buffer_load_dword")) {
            n = std::strstr(m, "x16") ? 16 : std::strstr(m, "x8") ? 8 : std::strstr(m, "x4") ? 4 : std::strstr(m, "x2") ? 2 : 1;
        } else if (!starts(m, "s_dcache_inv")) {
            n = 2;  // s_memtime, and anything unnamed
        }
        for (int k = 0; k < n; ++k) out.push_back(in.sdst + k);
        break;
    }
    case Enc::VOPC:
        pair(kVccLo);
        if (starts(m, "v_cmpx")) pair(kExecLo);
        break;
    case Enc::VOP3:
        if (in.op < 0x100) {  // a compare: its SGPR pair
            pair(in.sdst);
            if (starts(m, "v_cmpx")) pair(kExecLo);
        } else if (starts(m, "v_readlane") || starts(m, "v_readfirstlane")) {
            out.push_back(in.dst);
        } else if ((in.op >= 0x100 + 37 && in.op <= 0x100 + 42) || in.op == 0x16d || in.op == 0x16e) {
            pair(in.sdst);  // VOP3b: a carry-out or div_scale's
        }
        break;
    case Enc::VOP2:
        if (starts(m, "v_readlane")) out.push_back(in.dst);
        if (in.op >= 37 && in.op <= 42) pair(kVccLo);  // the carry ops
        break;
    case Enc::VOP1:
        if (starts(m, "v_readfirstlane")) out.push_back(in.dst);
        break;
    default:
        break;
    }
}

bool writes(const Inst& in, int reg) {
    static thread_local std::vector<int> w;
    scalar_writes(in, w);
    return std::find(w.begin(), w.end(), reg) != w.end();
}

bool is_branch(const Inst& in) {
    const char* m = mnemonic(in);
    return starts(m, "s_branch") || starts(m, "s_cbranch") || starts(m, "s_setpc") || starts(m, "s_swappc");
}

// The last instruction before `at` that writes `reg`, with no branch between
// it and `at`; -1 when there is a branch first or none writes it.
int last_writer(const Program& p, int at, int reg) {
    for (int i = at - 1; i >= 0; --i) {
        if (writes(p.insts[i], reg)) return i;
        if (is_branch(p.insts[i])) return -1;
    }
    return -1;
}

}  // namespace

GsEmitGate gs_emit_gate(const gcn::Program& p, std::uint32_t user_sgprs) {
    const int n = static_cast<int>(p.insts.size());
    std::vector<int> emits;
    for (int i = 0; i < n; ++i) {
        const Inst& in = p.insts[i];
        if (!starts(mnemonic(in), "s_sendmsg") || (in.imm & 0xf) != 2) continue;  // GS messages
        const int op = (in.imm >> 4) & 3;
        if (op == 3 || ((in.imm >> 8) & 3) != 0) return {};  // an emit-cut, or a stream past 0
        if (op == 2) emits.push_back(i);
    }
    for (int si = 0; si < n; ++si) {
        const Inst& sv = p.insts[si];
        if (!starts(mnemonic(sv), "s_and_saveexec_b64") || sv.src0 != kVccLo || sv.dst >= 104) continue;
        // The region: up to the s_mov_b64 that gives EXEC the saved mask back.
        int mi = -1;
        for (int i = si + 1; i < n && mi < 0; ++i) {
            const Inst& in = p.insts[i];
            if (starts(mnemonic(in), "s_mov_b64") && in.dst == kExecLo && in.src0 == sv.dst) mi = i;
            else if (writes(in, sv.dst) || writes(in, sv.dst + 1)) break;
        }
        if (mi < 0) continue;
        std::uint32_t inside = 0, outside = 0;
        for (const int e : emits) (e > si && e < mi ? inside : outside) += 1;
        if (!inside || !outside) continue;
        // VCC = a lane mask AND (sN > 0).
        const int ai = last_writer(p, si, kVccLo);
        if (ai < 0) continue;
        const Inst& a = p.insts[ai];
        if (!starts(mnemonic(a), "s_and_b64") || a.dst != kVccLo) continue;
        if (!((a.src0 == kVccLo && a.src1 < 104) || (a.src1 == kVccLo && a.src0 < 104))) continue;
        const int ci = last_writer(p, ai, kVccLo);
        if (ci < 0) continue;
        const Inst& c = p.insts[ci];
        if (c.enc != Enc::VOP3 || std::strcmp(mnemonic(c) ? mnemonic(c) : "", "v_cmp_gt_i32") != 0 || c.sdst != kVccLo) continue;
        if (c.src0 >= 104 || c.src1 != kConst0) continue;
        const int sn = c.src0;
        const int li = last_writer(p, ci, sn);
        if (li < 0) continue;
        const Inst& l = p.insts[li];
        if (l.enc != Enc::SMRD || !starts(mnemonic(l), "s_buffer_load_dword") || !l.imm_flag) continue;
        const int count = std::strstr(mnemonic(l), "x16") ? 16 : std::strstr(mnemonic(l), "x8") ? 8 : std::strstr(mnemonic(l), "x4") ? 4
                          : std::strstr(mnemonic(l), "x2") ? 2 : 1;
        if (sn < l.sdst || sn >= l.sdst + count) continue;
        // The V#: user data nothing has written before the load.
        const int vb = l.src0;
        if (vb % 4 != 0 || static_cast<std::uint32_t>(vb + 4) > user_sgprs) continue;
        bool clobbered = false;
        for (int i = 0; i < li && !clobbered; ++i) {
            for (int r = vb; r < vb + 4; ++r) clobbered = clobbered || writes(p.insts[i], r);
        }
        if (clobbered) continue;
        GsEmitGate g;
        g.vertices = outside;
        g.vsharp_sgpr = static_cast<std::uint32_t>(vb);
        g.dword = static_cast<std::uint32_t>(l.imm + (sn - l.sdst));
        g.gated = inside;
        return g;
    }
    return {};
}

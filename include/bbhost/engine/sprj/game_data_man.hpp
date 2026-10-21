// GameDataMan: session player records, save summaries, options and play data.
#pragma once

#include "bbhost/engine/base.hpp"

namespace bb {

inline constexpr Rva GAME_DATA_MAN_SINGLETON{0x58fb130};
// Both singleton creation sites allocate 0xc8 bytes with alignment 8.
// GameDataMan below intentionally remains a prefix view, not an owning type.
inline constexpr std::size_t GAME_DATA_MAN_ALLOCATION_SIZE = 0xc8;
inline constexpr Rva GAME_DATA_MAN_CONSTRUCT_FN{0x14e4e80};
inline constexpr Rva GAME_DATA_MAN_DESTRUCT_FN{0x14e5a80};
inline constexpr Rva GAME_DATA_MAN_CREATE_SINGLETON_FN{0x14e77d0};
inline constexpr Rva GAME_DATA_MAN_DESTROY_SINGLETON_FN{0x14e7820};
// Native accumulator multiplies its float delta by 1,000, truncates, adds in
// 32 bits, then clamps to PLAY_TIME_CAP. Its active caller is unresolved.
inline constexpr Rva GAME_DATA_MAN_ACCUMULATE_PLAY_TIME_FN{0x14eb6a0};
inline constexpr Rva GAME_DATA_MAN_SET_PLAY_TIME_FN{0x14eb720};
inline constexpr Rva GAME_DATA_MAN_GET_PLAY_TIME_DIV_1000_FN{0x14eb740};
inline constexpr std::uint32_t GAME_DATA_MAN_PLAY_TIME_CAP = 0xd693a018;
inline constexpr Rva GAME_DATA_MAN_ADD_HELP_WHITE_GHOST_COUNT_FN{0x14eb5c0};
inline constexpr Rva GAME_DATA_MAN_ADD_KILL_BLACK_GHOST_COUNT_FN{0x14eb5f0};
inline constexpr Rva GAME_DATA_MAN_ADD_TRUE_DEATH_COUNT_FN{0x14eb620};
inline constexpr Rva GAME_DATA_MAN_ADD_DEATH_COUNT_FN{0x14eb660};
inline constexpr Rva PLAYER_GAME_DATA_CONSTRUCT_FN{0x14f1510};
inline constexpr Rva PLAYER_GAME_DATA_REGISTER_DEBUG_MENU_FN{0x14f3e40};
// Removes debug-menu bindings only; does not destroy the PlayerGameData object.
inline constexpr Rva PLAYER_GAME_DATA_UNREGISTER_DEBUG_MENU_FN{0x14f21c0};
inline constexpr std::size_t OPTION_DATA_SIZE = 0x80;
inline constexpr Rva OPTION_DATA_CONSTRUCT_FN{0x14ee9c0};
// Resets selected options, then loads control/display/audio configuration.
// Preserves the DLC byte and the unresolved dword at +0x14.
inline constexpr Rva OPTION_DATA_RESET_FROM_CONFIGURATION_FN{0x14eea90};
// Transfers the whole 0x80-byte object through stream vfunc +0x20. The
// inspected function alone does not establish the stream's direction.
inline constexpr Rva OPTION_DATA_TRANSFER_RAW_FN{0x14ef080};
// Reads through stream vfunc +0x20, updates localization globals, and
// normalizes the inline UTF-16 text even if the transfer was short.
inline constexpr Rva OPTION_DATA_READ_AND_NORMALIZE_FN{0x14ef0b0};
inline constexpr Rva OPTION_DATA_APPLY_LOCALIZATION_FN{0x14ef420};
inline constexpr Rva OPTION_DATA_IMPORT_SETTINGS_FN{0x14ef550};
inline constexpr Rva OPTION_DATA_EXPORT_SETTINGS_FN{0x14ef730};
inline constexpr Rva OPTION_DATA_REGISTER_DEBUG_MENU_FN{0x14ef870};
inline constexpr std::size_t GAME_DATA_CHUNK_STORE_SIZE = 0x848;
inline constexpr Rva GAME_DATA_CHUNK_STORE_CONSTRUCT_FN{0x175aa30};
inline constexpr Rva GAME_DATA_CHUNK_STORE_DESTRUCT_FN{0x175abc0};
inline constexpr Rva GAME_DATA_CHUNK_STORE_WRITE_FN{0x175abe0};
// Skips unknown section IDs and mismatched payload sizes; its true return does
// not mean every stored section was recognized or applied.
inline constexpr Rva GAME_DATA_CHUNK_STORE_READ_FN{0x175ad70};
inline constexpr Rva GAME_DATA_FACE_SECTION_CONSTRUCT_FN{0x175afd0};
inline constexpr Rva GAME_DATA_FACE_SECTION_WRITE_FN{0x175b2d0};
inline constexpr Rva GAME_DATA_FACE_SECTION_READ_FN{0x175b650};
inline constexpr std::uint16_t GAME_DATA_FACE_SECTION_ID = 0xface;
inline constexpr std::size_t GAME_DATA_FACE_RECORD_COUNT = 8;
inline constexpr std::size_t GAME_DATA_FACE_RECORD_SIZE = 0xf4;
inline constexpr std::size_t GAME_DATA_FACE_WIRE_RECORD_SIZE = 0x100;
inline constexpr std::size_t GAME_DATA_FACE_SECTION_WIRE_SIZE = 0x800;
inline constexpr std::uint16_t GAME_DATA_CHUNK_END_ID = 0xffff;
// Ten character-save summaries, not the four multiplayer player-data rows.
inline constexpr std::size_t GAME_DATA_CHARACTER_SUMMARY_COUNT = 10;
inline constexpr std::size_t GAME_DATA_CHARACTER_SUMMARY_SIZE = 0x220;
inline constexpr std::size_t GAME_DATA_CHARACTER_SUMMARY_STORE_SIZE = 0x1560;
// Runtime save-summary fields used by host-side player identity.
inline constexpr std::size_t GAME_DATA_MAN_CHARACTER_SUMMARY_STORE_OFFSET = 0x58;
inline constexpr std::size_t GAME_DATA_MAN_PLAY_TIME_OFFSET = 0x94;
inline constexpr std::uint32_t GAME_DATA_CHARACTER_LEVEL_MAX = 544;
// The native GameDataMan prefix has no confirmed member names for these values.
inline constexpr std::size_t GAME_DATA_MAN_NG_CYCLE_OFFSET = 0x68;
inline constexpr std::size_t GAME_DATA_MAN_TRUE_DEATH_COUNT_OFFSET = 0x84;
inline constexpr std::size_t GAME_DATA_MAN_DEATH_COUNT_OFFSET = 0x88;
// Current field-by-field writer; older save versions use a different layout.
inline constexpr std::size_t GAME_DATA_CHARACTER_SUMMARY_WIRE_SIZE = 0x192;
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_STORE_CONSTRUCT_FN{0x14f6870};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_STORE_DESTRUCT_FN{0x14f8f60};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_STORE_WRITE_FN{0x14f9010};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_STORE_READ_FN{0x14f9100};
// Mutates one summary from the local player; does not set its occupancy byte.
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_UPDATE_LOCAL_FN{0x14f9490};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_GET_FN{0x14f9680};
// Sums the ten raw occupancy bytes, rather than counting nonzero entries.
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_SUM_OCCUPANCY_FN{0x14f96a0};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_FIND_EMPTY_FN{0x14f96e0};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_MARK_OCCUPIED_FN{0x14f9710};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_MARK_EMPTY_FN{0x14f9730};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_IS_OCCUPIED_FN{0x14f9750};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_WRITE_FN{0x14f6360};
inline constexpr Rva GAME_DATA_CHARACTER_SUMMARY_READ_FN{0x14f6600};
inline constexpr Rva PLAYER_GAME_DATA_COPY_SUMMARY_APPEARANCE_FN{0x14f3b40};
// Converts instance-form item handles before/after summary equipment copy. It
// preserves an encoded item identity, rather than simply zeroing the slot.
inline constexpr Rva GAITEM_REFERENCE_TO_NON_INSTANCE_FN{0x1a88050};
inline constexpr Rva GAITEM_REFERENCE_COPY_CONSTRUCT_FN{0x1a8bb70};
inline constexpr Rva GAITEM_REFERENCE_MOVE_ASSIGN_FN{0x1a8bcb0};
inline constexpr Rva GAITEM_REFERENCE_COPY_ASSIGN_FN{0x1a8bd80};
inline constexpr Rva GAITEM_REFERENCE_RELEASE_FN{0x1a8bc00};
inline constexpr Rva FACE_DATA_READ_FN{0x14e2560};
inline constexpr Rva PLAYER_EQUIPMENT_DATA_ASSIGN_FN{0x14d7f60};
inline constexpr Rva PLAYER_EQUIPMENT_DATA_READ_FN{0x1895a00};
inline constexpr Rva PLAYER_EQUIPMENT_DATA_DESTRUCT_FN{0x1893990};
// Proven prefix through the four native slot-flag records. Not the full native
// allocation size; the trailing manager range remains unmodeled.
inline constexpr std::size_t GAME_DATA_MAN_KNOWN_PREFIX_SIZE = 0xb1;

inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORDS_OFFSET = 0x10;
inline constexpr std::size_t GAME_DATA_MAN_BLOOD_MARK_ACTIVE_OFFSET = 0x28;
inline constexpr std::size_t GAME_DATA_MAN_BLOOD_MARK_DATA_OFFSET = 0x30;
inline constexpr std::size_t GAME_DATA_MAN_BLOOD_MARK_COLLISION_ID_OFFSET = 0x38;
inline constexpr std::size_t GAME_DATA_MAN_OCCUPIED_SLOTS_OFFSET = 0x18;
inline constexpr std::size_t GAME_DATA_MAN_SESSION_PLAYER_RECORDS_OFFSET = 0x20;
inline constexpr std::size_t GAME_DATA_MAN_FULL_RECOVER_REQUEST_OFFSET = 0x70;
inline constexpr std::size_t GAME_DATA_MAN_HELP_WHITE_GHOST_COUNT_OFFSET = 0x78;
inline constexpr std::size_t GAME_DATA_MAN_KILL_BLACK_GHOST_COUNT_OFFSET = 0x7c;
inline constexpr std::size_t GAME_DATA_MAN_SLOT_FLAGS_OFFSET = 0xa9;

inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SLOT_COUNT = 4;
// Event 0x0e has reserved this physical remote-player row.
inline constexpr std::uint8_t GAME_DATA_MAN_SLOT_STATE_ALLOCATED = 0x01;
// Packet 0x08 populated the row's primary player/game-data payload.
inline constexpr std::uint8_t GAME_DATA_MAN_SLOT_STATE_PLAYER_DATA_READY = 0x02;
// Packet 0x0c populated the first auxiliary peer payload.
inline constexpr std::uint8_t GAME_DATA_MAN_SLOT_STATE_AUXILIARY_DATA_0C_READY = 0x04;
// Packet 0x0d populated the second auxiliary peer payload.
inline constexpr std::uint8_t GAME_DATA_MAN_SLOT_STATE_AUXILIARY_DATA_0D_READY = 0x08;
// Packet 0x0b populated GameDataManPlayerRecord::character_type_or_init_param.
inline constexpr std::uint8_t GAME_DATA_MAN_SLOT_STATE_CHR_INIT_PARAM_READY = 0x10;
// Pointer-table length scanned by WorldChrMan_AllocateRemotePlayerIns after
// the local row and four ordinary remote rows.
inline constexpr std::size_t GAME_DATA_MAN_SESSION_PLAYER_RECORD_COUNT = 0x28;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_SIZE = 0x6a0;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_MODEL_ID_OFFSET = 0x10;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_STATS_OFFSET = 0x40;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_LEVEL_OFFSET = 0x90;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_BLOOD_ECHOES_OFFSET = 0x94;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_CHR_INIT_PARAM_OFFSET = 0xa4;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_SERVER_USER_ID_OFFSET = 0x688;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_SERVER_CHR_ID_OFFSET = 0x690;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_SESSION_DESCRIPTOR_INDEX_OFFSET = 0x5d8;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_RECORD_NAT_TYPE_OFFSET = 0x698;

inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_VITALITY_OFFSET = 0x00;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_ENDURANCE_OFFSET = 0x08;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_STRENGTH_OFFSET = 0x18;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_SKILL_OFFSET = 0x20;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_BLOODTINGE_OFFSET = 0x28;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_ARCANE_OFFSET = 0x30;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_STATS_INSIGHT_OFFSET = 0x44;

inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_OFFSET = 0x30;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_VITALITY_OFFSET = 0x00;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_ENDURANCE_OFFSET = 0x04;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_STRENGTH_OFFSET = 0x0c;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_SKILL_OFFSET = 0x10;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_BLOODTINGE_OFFSET = 0x14;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_ARCANE_OFFSET = 0x18;
inline constexpr std::size_t GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_LEVEL_OFFSET = 0x28;

// Native recoverable Blood Echoes record, not a network player allocation.
struct BloodMarkData {
    float placement_position[3];
    float placement_rotation[3];
    float death_position[3];
    float death_rotation[3];
    // Commit initializes this to -1; later ownership/meaning untraced.
    std::int32_t unknown_30;
    Unknown<0x04> _unk034;
    std::uint32_t blood_echoes;
    std::uint32_t map_id;
    // Decoded from seven altar-selection event bits beginning at 9020. Native
    // sets -1 outside an m29 map.
    std::int32_t chalice_altar_index;
    // Nonzero follows native alternate/curse death handling.
    std::uint8_t alternate_death;
    Unknown<0x03> _unk045;
};

// Versioned FACE payload copied by the native chunk serializer, size 0xf0. The
// constructor writes magic FACE, version 3, size 0xf0. Individual face
// parameters remain opaque; this is not a live FaceGen object with a vtable.
struct GameDataFacePayload {
    std::uint8_t magic[4];
    std::uint32_t version;
    std::uint32_t size;
    std::uint8_t parameters[0xe4];
};

// Native debug-menu name FaceData; embedded at PlayerGameData+0x498 and
// character summary+0x38. This 0x118-byte runtime view includes a vtable; only
// the dword at +0x10 and FACE payload are written by the summary writer.
struct FaceData {
    const void* vtable;
    // Copied from local PlayerGameData+0x4a0; its ownership is unresolved.
    std::uint64_t unknown08;
    std::int32_t unknown10;
    GameDataFacePayload face;
    Unknown<0x14> _unk104;
};

// Equipment subobject view shared by PlayerGameData+0x240 and summary+0x150.
// Descriptive name, not a recovered native class name. The sixteen runtime
// reference words require native assignment/release, not bitwise cloning.
// They can encode instance-backed references OR non-instance IDs. The current
// save reader rebuilds them from the separate serialized IDs.
struct PlayerEquipmentData {
    const void* vtable;
    std::uint32_t unknown08;
    std::uint32_t saved_fields[6];
    std::uint32_t runtime_item_references[16];
    std::uint32_t saved_item_ids[16];
    // Native assignment copies through +0xbb, but the current serializer does
    // not transfer these bytes. The trailing four bytes remain opaque.
    Unknown<0x1c> _unka4;
};

// Character-save summary, not a PlayerIns or full PlayerGameData snapshot.
// Store+0x20 contains ten such records. Native updating and loading have
// reference-management side effects; do not copy these bitwise.
struct GameDataCharacterSummary {
    // Native updater copies seventeen UTF-16 units; loading forces unit 16 to
    // NUL and normalizes native menu glyph aliases, even after a short read.
    std::uint16_t character_name[17];
    std::uint8_t _reserved22[2];
    std::uint32_t level;
    std::uint32_t play_time_div_1000;
    // Only the low dword of PlayerGameData.total_blood_echoes_acquired.
    std::uint32_t total_blood_echoes_acquired_low;
    // Captured from the active world-resource entry; NOT a respawn ID. Updater
    // can write 0xffffffff if that lookup fails.
    std::uint32_t map_id;
    // Positive value from singleton RVA 0x5562878+0x1624, else 0x96b.
    std::int32_t unknown34;
    FaceData face_data;
    PlayerEquipmentData equipment_data;
    // Copied from PlayerGameData+0xca; meaning not established.
    std::uint8_t player_byte_ca;
    std::uint8_t archetype;
    std::uint8_t gift;
    std::uint8_t appearance;
    std::uint8_t face_type;
    std::uint8_t hair_type;
    std::uint8_t hair_eyes_color;
    // Comparison result CSPlaygo+0x48 == 0, stored as a native byte.
    std::uint8_t playgo_state_is_zero;
    // Copied from singleton RVA 0x5562878+0x162c.
    std::uint32_t unknown218;
    std::uint8_t _reserved21c[4];
};

// GameDataMan+0x58 owned allocation. Semantic name established from native
// character-save callers; original C++ class name remains unresolved.
struct GameDataCharacterSummaryStore {
    const void* vtable;
    Unknown<0x08> _unk08;
    // Separately managed by save/delete callers. Clearing occupancy does not
    // clear a record; the writer still serializes every record.
    std::uint8_t occupied[GAME_DATA_CHARACTER_SUMMARY_COUNT];
    std::uint8_t _reserved1a[6];
    GameDataCharacterSummary records[GAME_DATA_CHARACTER_SUMMARY_COUNT];
};

// In-memory face slot, stride 0xf4. Native serialization canonicalizes any
// nonzero populated flag to 1.
struct GameDataFaceRecord {
    std::uint8_t populated;
    std::uint8_t unknown_flag;
    std::uint8_t _reserved02[2];
    GameDataFacePayload face;
};

// Serialized face record, stride 0x100, NOT the runtime slot layout.
// Unpopulated slots are entirely zeroed by the native writer. For populated
// slots, no initialization of reserved02 is established by that writer.
struct GameDataFaceWireRecord {
    std::uint8_t populated;
    std::uint8_t unknown_flag;
    std::uint8_t reserved02[0x0e];
    GameDataFacePayload face;
};

// Embedded at chunk store +0x10. The fixed vector has eight runtime slots; its
// count is initialized/rebuilt to eight, including unpopulated records.
struct GameDataFaceSection {
    const void* vtable;
    GameDataFaceRecord records[GAME_DATA_FACE_RECORD_COUNT];
    std::uint8_t _reserved7a8[8];
    std::size_t record_count;
};

// Section ID 1, embedded at chunk store +0x7c8. Native read/write copies only
// payload (+8..+0x27). The following ten words are initialized to 0x80000069
// but are not included by those methods; their meanings remain unresolved.
struct GameDataSectionOne {
    const void* vtable;
    std::uint8_t payload[0x20];
    std::uint32_t unknown_words[10];
};

// Eight-byte header used by the chunk reader/writer. Each header is aligned to
// four bytes. The payload length excludes this header; 0xffff terminates.
struct GameDataChunkHeader {
    std::uint16_t section_id;
    std::uint16_t reserved;
    std::uint32_t payload_size;
};

// GameDataMan+0x50 allocation, size 0x848. Descriptive name: native class
// spelling is not established. All section storage is embedded; section
// pointers borrow those subobjects. This runtime object contains
// vtables/self-pointers and cannot be serialized by memcpy; use the native
// chunk format.
struct GameDataChunkStore {
    const void* vtable;
    std::uint16_t unknown08;
    std::uint16_t unknown0a;
    // Initialized to 0x1000; purpose is not established by the serializers.
    std::uint32_t unknown0c;
    GameDataFaceSection face_section;
    GameDataSectionOne section_one;
    // Fixed capacity four, initialized with only the two embedded sections.
    void* sections[4];
    std::uint8_t _reserved838[8];
    std::size_t section_count;
};

// Native OptionData owned by GameDataMan+0x48, with no vtable.
//
// Names through +0x12 are established by the native debug menu at RVA
// 0x14ef870, except the deliberately opaque +0x11. Byte flags stay u8: native
// deserialization does not validate every byte as a bool. This is the
// saved/runtime representation, not the differently laid-out settings object
// consumed by RVA 0x14ef550. Changing this memory is not an apply API.
struct OptionData {
    std::uint8_t camera_speed;
    std::uint8_t pad_vibration;
    std::uint8_t brightness;
    std::uint8_t sound_type;
    std::uint8_t bgm_volume;
    std::uint8_t se_volume;
    std::uint8_t voice_volume;
    std::uint8_t blood_level;
    std::uint8_t caption_visible;
    // Native label is FE Visible; no broader rendering meaning is assumed.
    std::uint8_t fe_visible;
    std::uint8_t camera_axis_side;
    std::uint8_t camera_axis_inverse;
    std::uint8_t lock_on_auto_switch;
    std::uint8_t auto_avoidance_wall;
    std::uint8_t enable_rank_register;
    std::uint8_t rank_register_profile_index;
    std::uint8_t allow_global_matching;
    std::uint8_t _unk11;
    std::uint8_t unlocked_dlc_flag;
    // Cleared by reset and after raw loading; exact role not established.
    std::uint8_t load_reset_byte;
    Unknown<0x04> _unk14;
    // -1 selects system-language resolution. Exact user-facing category and
    // enum names remain unresolved; the result updates global RVA 0x55992f0.
    std::int32_t localization_selection_18;
    // -1 also resolves from system language, via RVA 0x1fb97e0; the result
    // updates global RVA 0x5128954 through a different regional lookup table.
    std::int32_t localization_selection_1c;
    // Import copies at most eight UTF-16 code units and writes the ninth as
    // NUL. Loading normalizes punctuation. Its UI purpose is unconfirmed.
    std::uint16_t settings_text[9];
    Unknown<0x4e> _unk32;
};

// Two per-slot latches cleared with the payload-state byte on peer leave. The
// first gates native peer-data reuse during member retirement. The second
// records that the matching remote actor has been observed/processed. Their
// stock names are not in the binary; the names are behavioral.
struct GameDataManPlayerSlotFlags {
    std::uint8_t data_ready;
    std::uint8_t world_chr_ready;
};

struct GameDataManPlayerStats {
    std::uint32_t vitality;
    std::uint32_t _unk44;
    std::uint32_t endurance;
    std::uint32_t _unk4c;
    std::uint32_t _unk50;
    std::uint32_t _unk54;
    std::uint32_t strength;
    std::uint32_t _unk5c;
    std::uint32_t skill;
    std::uint32_t _unk64;
    std::uint32_t bloodtinge;
    std::uint32_t _unk6c;
    std::uint32_t arcane;
    std::uint32_t _unk74;
    std::uint32_t _unk78;
    std::uint32_t _unk7c;
    std::uint32_t _unk80;
    std::uint32_t insight;
    std::uint32_t _unk88;
};

// Stat block as observed on the subtype-0x08 wire payload.
struct GameDataManPlayerSubtype08WireStats {
    std::uint32_t vitality;
    std::uint32_t endurance;
    std::uint32_t _unk08;
    std::uint32_t strength;
    std::uint32_t skill;
    std::uint32_t bloodtinge;
    std::uint32_t arcane;
    Unknown<0x0c> _unk1c;
    std::uint32_t level;
};

// Session-visible player data consumed by CSMultiPlayerInsTask::STEP_Create.
// Native debug-menu class name: PlayerGameData (RVA 0x14f21c0/0x14f3e40).
//
// The model id and chr-init-param offsets are confirmed by RVA 0x1e50090. The
// stat block is confirmed by live 1.09 reads and the player-game-data debug
// menu. Level is copied into record +0x90; local/main Blood Echoes live at
// record +0x94.
struct GameDataManPlayerRecord {
    static constexpr std::int32_t UNASSIGNED_MODEL_ID = -1;

    Unknown<0x10> _unk00;
    // Native menu calls this PlayerNo; retained name matches existing callers.
    std::int32_t model_id;
    // Saved/session data values, not live ChrDataModule health authority.
    std::int32_t hp;
    std::int32_t max_hp;
    std::int32_t base_max_hp;
    // Native legacy menu labels MP/MaxMP; no new Bloodborne mechanic implied.
    std::int32_t mp;
    std::uint32_t _unk24;
    std::int32_t max_mp;
    std::uint32_t _unk2c;
    std::int32_t stamina;
    std::int32_t max_stamina;
    std::int32_t base_max_stamina;
    std::uint32_t _unk3c;
    GameDataManPlayerStats stats;
    std::uint32_t _unk8c;
    std::uint32_t level;
    std::uint32_t blood_echoes;
    // Native menu label: Total Get Soul.
    std::uint64_t total_blood_echoes_acquired;
    std::int32_t total_added_parameters;
    std::uint32_t character_type_or_init_param;
    Unknown<0x23> _unk0a8;
    std::uint8_t voice_type;
    std::uint16_t shop_level;
    std::uint8_t archetype;
    std::uint8_t appearance;
    std::uint8_t gift;
    Unknown<0x03> _unk0d1;
    std::uint32_t multiplayer_count;
    std::uint16_t coop_success_count;
    std::uint16_t coop_failure_count;
    std::uint16_t oppose_success_count;
    std::uint16_t oppose_failure_count;
    Unknown<0x20> _unk0e0;
    std::int32_t poison_resistance;
    std::int32_t blood_resistance;
    std::int32_t disease_resistance;
    std::int32_t curse_resistance;
    std::int32_t therianthrope_resistance;
    Unknown<0x04> _unk114;
    std::uint8_t face_type;
    std::uint8_t hair_type;
    std::uint8_t hair_eyes_color;
    std::uint8_t curse_level;
    std::uint8_t invade_type;
    Unknown<0x123> _unk11d;
    // Shared subobject copied into character-save summaries by RVA 0x14f3b40.
    PlayerEquipmentData equipment_data;
    Unknown<0x198> _unk300;
    // Native debug menu names this embedded object FaceData.
    FaceData face_data;
    // Allocated only for local records; native debug class RepositoryData.
    Unknown<0xa0>* repository_data;
    // Allocated only for local records; native debug class GestureGameData.
    Unknown<0x88>* gesture_game_data;
    Unknown<0x18> _unk5c0;
    // Index into the 0x22-entry session descriptor table used for multiplayer
    // title, message, team, and summon presentation data.
    std::uint8_t session_descriptor_index;
    Unknown<0xaf> _unk5d9;
    // Native menu label ServerUserId; constructor sentinel 0x8000000000000000.
    // Scalar server identity, not an owned pointer or the mod's NPID table.
    std::uint64_t server_user_id;
    std::uint64_t server_chr_id;
    // Native menu label NatType (0..3), not a timestamp or activity counter.
    std::uint32_t nat_type;
    Unknown<0x04> _unk69c;

    bool is_assigned() const { return model_id != UNASSIGNED_MODEL_ID; }
};

// Bloodborne's session-visible player record array owner. Remote records are
// addressed as *(GameDataMan + 0x10) + slot * 0x6a0. Slot occupancy bytes at
// *(GameDataMan + 0x18) gate the member-slot scan in RVA 0x178b5e0 case 0xe.
struct GameDataMan {
    // Constructor allocates 0x48 bytes; destructor names it TrophyEquipData.
    Unknown<0x48>* trophy_equip_data;
    GameDataManPlayerRecord* local_player_record;
    // Four records inside a 0x1a90-byte allocation. This points 0x10 bytes
    // past the allocation base; the preceding header stores base and count.
    GameDataManPlayerRecord* player_records;
    std::uint8_t* occupied_slots;
    // Owns each non-null record and the separately allocated pointer table.
    GameDataManPlayerRecord** session_player_records;
    Unknown<0x08> _unk28;
    // Blood-mark save record populated by RVA 0x13d2910. Legacy field name
    // retained; the separate +0x40 object's purpose remains unresolved.
    BloodMarkData* unknown_data_30;
    Unknown<0x08> _unk38;
    Unknown<0x48>* unknown_data_40;
    // Named OptionData by the native destructor's debug-menu removal.
    OptionData* option_data;
    // Chunked data with an eight-slot FACE section and a section numbered 1.
    // The native class name remains unresolved; legacy field name retained.
    GameDataChunkStore* unknown_data_50;
    // Character-save summaries. Native class name unresolved; historical field
    // name retained. Not session-slot or live-actor storage.
    GameDataCharacterSummaryStore* unknown_data_58;
    // Named PcOptionData by the destructor; null in this constructor.
    void* pc_option_data;
    // Native PRIMAL_GAME debug-menu field: ClearCount.
    std::int32_t clear_count;
    // Native menu enumerates none=0, good=1, bad=2.
    std::int32_t clear_state;
    std::uint8_t full_recover_request;
    Unknown<0x03> _unk71;
    // Native PRIMAL_GAME debug-menu field: ItemComplete.
    std::uint32_t item_complete;
    // Historical signed view. Native add helper performs unsigned saturated
    // addition, with 0xffffffff as the maximum, not exclusively an invalid ID.
    std::int32_t help_white_ghost_count;
    // Same unsigned saturation behavior as help_white_ghost_count.
    std::int32_t kill_black_ghost_count;
    std::uint8_t true_death;
    Unknown<0x03> _unk81;
    std::uint32_t true_death_count;
    std::uint32_t death_count;
    Unknown<0x08> _unk8c;
    // Native menu names this m_playTime. Accumulator multiplies a supplied
    // delta by 1,000; summary/getter divide this field by 1,000. Milliseconds
    // are consistent with the cap, but the active delta producer is unresolved.
    std::uint32_t play_time_raw;
    // Native accumulator subtracts one (clamped 0..2000) when decay_elapsed
    // exceeds 10 delta units. Gameplay meaning remains unassigned.
    std::int32_t decaying_counter;
    // Set to zero, discarding excess, after each counter decrement. Updated
    // independently of the integer play-time cap; not a packet retry timer.
    float decay_elapsed;
    // Included by the global save-data writer/reader at RVA 0x14eb350/0x14eb470.
    std::uint32_t unknown_a0;
    std::uint32_t unknown_a4;
    // Native getter RVA 0x14eb7f0 tests this for nonzero.
    std::uint8_t unknown_a8;
    GameDataManPlayerSlotFlags slot_flags[GAME_DATA_MAN_PLAYER_SLOT_COUNT];

    // One of the four physical remote-player rows, or null.
    GameDataManPlayerRecord* player_record_ptr(std::size_t slot) const {
        if (slot >= GAME_DATA_MAN_PLAYER_SLOT_COUNT || !player_records) return nullptr;
        return player_records + slot;
    }
    // The native payload-state byte for one of the four physical remote-player
    // rows. Event 0x0e creates the row with bit 0x01; the packet pump adds bits
    // 0x02, 0x04, 0x08 and 0x10 as payloads arrive. Event 0x0f and full session
    // reset clear the byte to zero. False when out of range or unallocated.
    bool player_slot_state(std::size_t slot, std::uint8_t* out) const {
        if (slot >= GAME_DATA_MAN_PLAYER_SLOT_COUNT || !occupied_slots) return false;
        *out = occupied_slots[slot];
        return true;
    }
    // An auxiliary/session record from the native 40-entry pointer table, or
    // null. These participate in model-id resolution but are not part of the
    // four ordinary multiplayer slot rows.
    GameDataManPlayerRecord* session_player_record_ptr(std::size_t slot) const {
        if (slot >= GAME_DATA_MAN_SESSION_PLAYER_RECORD_COUNT || !session_player_records) return nullptr;
        return session_player_records[slot];
    }
    bool full_recover_requested() const { return full_recover_request != 0; }
    // Legacy filtered views of help_white_ghost_count and
    // kill_black_ghost_count (named apart because C++ cannot share a name with
    // the field). False also represents unsigned saturation at 0xffffffff; it
    // is not proof of missing game data.
    bool help_white_ghost_count_filtered(std::int32_t* out) const {
        if (help_white_ghost_count == -1) return false;
        *out = help_white_ghost_count;
        return true;
    }
    bool kill_black_ghost_count_filtered(std::int32_t* out) const {
        if (kill_black_ghost_count == -1) return false;
        *out = kill_black_ghost_count;
        return true;
    }
};

namespace detail::game_data_man_layout {
BB_SIZE(FaceData, 0x118);
BB_OFFSET(FaceData, face, 0x14);
BB_SIZE(GameDataFacePayload, 0xf0);
BB_SIZE(PlayerEquipmentData, 0xc0);
BB_OFFSET(PlayerEquipmentData, runtime_item_references, 0x24);
BB_OFFSET(PlayerEquipmentData, saved_item_ids, 0x64);
BB_SIZE(GameDataCharacterSummary, GAME_DATA_CHARACTER_SUMMARY_SIZE);
BB_OFFSET(GameDataCharacterSummary, level, 0x24);
BB_OFFSET(GameDataCharacterSummary, play_time_div_1000, 0x28);
BB_OFFSET(GameDataCharacterSummary, face_data, 0x38);
BB_OFFSET(GameDataCharacterSummary, equipment_data, 0x150);
BB_OFFSET(GameDataCharacterSummary, player_byte_ca, 0x210);
BB_OFFSET(GameDataCharacterSummary, unknown218, 0x218);
BB_SIZE(GameDataCharacterSummaryStore, GAME_DATA_CHARACTER_SUMMARY_STORE_SIZE);
BB_OFFSET(GameDataCharacterSummaryStore, occupied, 0x10);
BB_OFFSET(GameDataCharacterSummaryStore, records, 0x20);
static_assert(GAME_DATA_MAN_CHARACTER_SUMMARY_STORE_OFFSET == offsetof(GameDataMan, unknown_data_58), "character summary store offset");
static_assert(GAME_DATA_MAN_PLAY_TIME_OFFSET == offsetof(GameDataMan, play_time_raw), "manager play-time offset");
BB_SIZE(GameDataFaceRecord, GAME_DATA_FACE_RECORD_SIZE);
BB_SIZE(GameDataFaceWireRecord, GAME_DATA_FACE_WIRE_RECORD_SIZE);
BB_SIZE(GameDataChunkHeader, 8);
BB_SIZE(GameDataChunkStore, GAME_DATA_CHUNK_STORE_SIZE);
BB_OFFSET(GameDataChunkStore, face_section, 0x10);
BB_OFFSET(GameDataChunkStore, section_one, 0x7c8);
BB_SIZE(OptionData, OPTION_DATA_SIZE);

BB_OFFSET(GameDataMan, unknown_data_58, 0x58);
BB_OFFSET(GameDataMan, decaying_counter, 0x98);
BB_OFFSET(GameDataMan, decay_elapsed, 0x9c);
BB_OFFSET(GameDataMan, unknown_a0, 0xa0);
BB_OFFSET(GameDataMan, unknown_a4, 0xa4);
BB_OFFSET(GameDataMan, unknown_a8, 0xa8);
BB_OFFSET(GameDataMan, slot_flags, 0xa9);
static_assert(offsetof(GameDataMan, slot_flags) + sizeof(GameDataMan::slot_flags) == GAME_DATA_MAN_KNOWN_PREFIX_SIZE,
              "GameDataMan known prefix");
BB_SIZE(GameDataMan, 0xb8);
BB_OFFSET(GameDataMan, player_records, GAME_DATA_MAN_PLAYER_RECORDS_OFFSET);
BB_OFFSET(GameDataMan, occupied_slots, GAME_DATA_MAN_OCCUPIED_SLOTS_OFFSET);
BB_OFFSET(GameDataMan, session_player_records, GAME_DATA_MAN_SESSION_PLAYER_RECORDS_OFFSET);
BB_OFFSET(GameDataMan, unknown_data_30, GAME_DATA_MAN_BLOOD_MARK_DATA_OFFSET);
BB_OFFSET(GameDataMan, full_recover_request, GAME_DATA_MAN_FULL_RECOVER_REQUEST_OFFSET);
BB_OFFSET(GameDataMan, help_white_ghost_count, GAME_DATA_MAN_HELP_WHITE_GHOST_COUNT_OFFSET);
BB_OFFSET(GameDataMan, kill_black_ghost_count, GAME_DATA_MAN_KILL_BLACK_GHOST_COUNT_OFFSET);
BB_OFFSET(GameDataMan, slot_flags, GAME_DATA_MAN_SLOT_FLAGS_OFFSET);
BB_SIZE(GameDataManPlayerSlotFlags, 0x02);

BB_OFFSET(GameDataManPlayerRecord, equipment_data, 0x240);
BB_OFFSET(GameDataManPlayerRecord, face_data, 0x498);
BB_SIZE(GameDataManPlayerRecord, GAME_DATA_MAN_PLAYER_RECORD_SIZE);
BB_OFFSET(GameDataManPlayerRecord, model_id, GAME_DATA_MAN_PLAYER_RECORD_MODEL_ID_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, stats, GAME_DATA_MAN_PLAYER_RECORD_STATS_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, level, GAME_DATA_MAN_PLAYER_RECORD_LEVEL_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, blood_echoes, GAME_DATA_MAN_PLAYER_RECORD_BLOOD_ECHOES_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, character_type_or_init_param, GAME_DATA_MAN_PLAYER_RECORD_CHR_INIT_PARAM_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, session_descriptor_index,
          GAME_DATA_MAN_PLAYER_RECORD_SESSION_DESCRIPTOR_INDEX_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, nat_type, GAME_DATA_MAN_PLAYER_RECORD_NAT_TYPE_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, server_user_id, GAME_DATA_MAN_PLAYER_RECORD_SERVER_USER_ID_OFFSET);
BB_OFFSET(GameDataManPlayerRecord, server_chr_id, GAME_DATA_MAN_PLAYER_RECORD_SERVER_CHR_ID_OFFSET);

BB_OFFSET(GameDataManPlayerStats, vitality, GAME_DATA_MAN_PLAYER_STATS_VITALITY_OFFSET);
BB_OFFSET(GameDataManPlayerStats, endurance, GAME_DATA_MAN_PLAYER_STATS_ENDURANCE_OFFSET);
BB_OFFSET(GameDataManPlayerStats, strength, GAME_DATA_MAN_PLAYER_STATS_STRENGTH_OFFSET);
BB_OFFSET(GameDataManPlayerStats, skill, GAME_DATA_MAN_PLAYER_STATS_SKILL_OFFSET);
BB_OFFSET(GameDataManPlayerStats, bloodtinge, GAME_DATA_MAN_PLAYER_STATS_BLOODTINGE_OFFSET);
BB_OFFSET(GameDataManPlayerStats, arcane, GAME_DATA_MAN_PLAYER_STATS_ARCANE_OFFSET);
BB_OFFSET(GameDataManPlayerStats, insight, GAME_DATA_MAN_PLAYER_STATS_INSIGHT_OFFSET);

BB_SIZE(GameDataManPlayerSubtype08WireStats, 0x2c);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, vitality, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_VITALITY_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, endurance, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_ENDURANCE_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, strength, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_STRENGTH_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, skill, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_SKILL_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, bloodtinge,
          GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_BLOODTINGE_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, arcane, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_ARCANE_OFFSET);
BB_OFFSET(GameDataManPlayerSubtype08WireStats, level, GAME_DATA_MAN_PLAYER_SUBTYPE_08_WIRE_STATS_LEVEL_OFFSET);
}  // namespace detail::game_data_man_layout

}  // namespace bb
